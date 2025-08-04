import torch
import time
from torch.autograd import Variable
from utils.utils import AverageMeter
import random

def pretrain_epoch_contrastive(epoch, data_loader, model, criterion, optimizer, opt):
    print(f'Contrastive pretraining at epoch {epoch}')
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end_time = time.time()
    
    for i, (inputs, _) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        
        if not opt.no_cuda:
            inputs = [inp.cuda() for inp in inputs]
        
        inputs = [Variable(inp) for inp in inputs]
        
        idx1, idx2 = random.sample(range(len(inputs)), 2)
        features1 = model.module.forward_features(inputs[idx1])
        features2 = model.module.forward_features(inputs[idx2])
        loss = criterion(features1, features2)
        losses.update(loss.item(), inputs[0].size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 10 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}]\t'
                  f'lr: {optimizer.param_groups[0]["lr"]:.5f}\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')

def pretrain_epoch_class(epoch, data_loader, model, criterion, optimizer, opt):
    print(f'Class pretraining at epoch {epoch}')

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()
        
        targets = Variable(targets)
        
        for modality_idx, modality_input in enumerate(inputs):
            modality_input = Variable(modality_input)
            outputs = model([modality_input])
            loss = criterion(outputs, targets)
            
            losses.update(loss.data, modality_input.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]	 lr: {lr:.5f}	'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})	'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})	'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})	'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      lr=optimizer.param_groups[0]['lr']))

