import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from utils.opts import parse_opts, import_network
from utils.utils import *

from steps.pretrain import pretrain_epoch_class, pretrain_epoch_crossclr
from steps.train import train_epoch
from steps.validation import val_epoch
from steps.crossclr_loss import CrossCLR_onlyIntraModality, MaxMargin_coot
from steps.setup_loaders import get_train_setup, get_val_setup, get_test_setup
from steps.test import test

# from models.phinet3d_c3i import make_multi_input, generate_model
from models.resnet import make_multi_input, generate_model


if __name__ == '__main__':
    opt = parse_opts()

    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    make_multi_input, generate_model = import_network(opt.model)
    model, parameters = generate_model(opt)
    train_loader, train_logger, train_batch_logger = get_train_setup(opt)


    #### STEP 1: Pretrain with CrossCLR
    
    print('running pretraining for {} epochs'.format(opt.pretrain_epochs))
    # criterion_pretrain = CrossCLR_onlyIntraModality(temperature=0.03, negative_weight=0.8)
    criterion_pretrain = MaxMargin_coot(True) # for standard contrastive pretraining
    pretrain_optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=opt.dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)

    for i in range(opt.pretrain_epochs):
        pretrain_epoch_contrastive(i, train_loader, model, criterion_pretrain, pretrain_optimizer, opt)

    #### STEP 2: Pretrain with classification loss

    criterion_pretrain = nn.CrossEntropyLoss().cuda()
    pretrain_optimizer = optim.SGD(
        model.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=opt.dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    
    for i in range(opt.pretrain_epochs):
        pretrain_epoch_class(i, train_loader, model, criterion_pretrain, pretrain_optimizer, opt)
    del pretrain_optimizer

    # after pretraining, create the muti-input model from the single input one for fine-tuning
    phinet_multiinput = make_multi_input(model.module, len(opt.ehw_cam))
    model = nn.DataParallel(phinet_multiinput, device_ids=None)
    
    #### STEP 3: Train with multimodal inputs

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=opt.dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=opt.lr_patience)
    val_loader, val_logger, class_to_idx = get_val_setup(opt)

    best_prec1 = 0
    print('running main training')
    for i in range(opt.n_epochs + 1):

        adjust_learning_rate(optimizer, i, opt)
        train_epoch(i, train_loader, model, criterion, optimizer, opt,
                    train_logger, train_batch_logger)
        state = {
            'epoch': i,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1
            }
        save_checkpoint(state, False, opt)
            
        if not opt.no_val:
            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger, class_to_idx)
            if opt.random_mask_fraction > 0:
                val_epoch_single_modality(i, val_loader, model, criterion, opt, val_logger, class_to_idx)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, is_best, opt)

    test_loader, class_names = get_test_setup(opt)
    test(test_loader, model, opt, class_names)