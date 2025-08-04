import torch
from torch.autograd import Variable
import time
from utils.utils import *

def map_to_group(targets, class_to_idx, mapping):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    mapped_targets = torch.tensor([mapping.get(idx_to_class[t.item()], -1) for t in targets])
    return mapped_targets.to(targets.device)

def compute_group_accuracy(outputs, targets, class_to_idx, mapping):
    group_targets = map_to_group(targets, class_to_idx, mapping)
    group_preds = map_to_group(outputs.argmax(dim=1), class_to_idx, mapping)
    valid_mask = group_targets != -1
    if valid_mask.any():
        return (group_preds[valid_mask] == group_targets[valid_mask]).float().mean() * 100
    return torch.tensor(0.0, device=targets.device)

def val_epoch(epoch, data_loader, model, criterion, opt, logger, class_to_idx):
    print('validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_action = AverageMeter()
    top1_quality = AverageMeter()

    # Define class merging for action and quality
    action_mapping = {"FTF": 0, "FTN": 0, "FTS": 0,
                      "OCF": 1, "OCN": 1, "OCS": 1,
                      "PSF": 2, "PSN": 2, "PSS": 2,
                      "NOSE": 3, "TR": 4}

    quality_mapping = {"FTF": 0, "OCF": 0, "PSF": 0,  # F
                       "FTN": 1, "OCN": 1, "PSN": 1,  # N
                       "FTS": 2, "OCS": 2, "PSS": 2}  # S

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()
        with torch.no_grad():
            inputs = [Variable(inp) for inp in inputs]
            targets = Variable(targets)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        inputs = inputs[0]
        
        # Joint gesture - quality classification
        prec1 = calculate_accuracy(outputs.data, targets.data, topk=(1,))
        top1.update(prec1[0], inputs.size(0))
        
        # Gesture classification from single classification head (can be optimized with a single task loss)
        action_acc = compute_group_accuracy(outputs, targets, class_to_idx, action_mapping)
        top1_action.update(action_acc.item(), inputs.size(0))

        # Quality classification from single classification head (as before)
        quality_acc = compute_group_accuracy(outputs, targets, class_to_idx, quality_mapping)
        top1_quality.update(quality_acc.item(), inputs.size(0))
        
        losses.update(loss.data, inputs.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Action@1 {top1_action.val:.5f} ({top1_action.avg:.5f})\t'
              'Quality@1 {top1_quality.val:.5f} ({top1_quality.avg:.5f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  top1_action=top1_action,
                  top1_quality=top1_quality))

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'action_prec1': top1_action.avg,
                'quality_prec1': top1_quality.avg})

    return losses.avg.item(), top1.avg.item()