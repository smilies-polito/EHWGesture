import os
import torch
from transforms.spatial_transforms import Normalize, MultiScaleRandomCrop, MultiScaleCornerCrop, Compose, RandomHorizontalFlip, Scale, CenterCrop, CornerCrop, ToTensor
from transforms.temporal_transforms import TemporalRandomCrop, TemporalCenterCrop
from transforms.target_transforms import ClassLabel, VideoID
from datasets.dataset import get_training_set, get_validation_set, get_test_set
from utils.utils import Logger

def get_train_setup(opt):
    norm_method = Normalize([0, 0, 0] if opt.no_mean_norm else opt.mean, [1, 1, 1] if not opt.std_norm else opt.std)

    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c'])

    spatial_transform = Compose([
        RandomHorizontalFlip(),
        crop_method,
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform, ehw_cam=opt.ehw_cam, ehw_input=opt.ehw_input)
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_threads, pin_memory=True)

    train_logger = Logger(os.path.join(opt.result_path, 'train.log'), ['epoch', 'loss', 'prec1', 'lr'])
    train_batch_logger = Logger(os.path.join(opt.result_path, 'train_batch.log'), ['epoch', 'batch', 'iter', 'loss', 'prec1', 'lr'])

    return train_loader, train_logger, train_batch_logger

def get_val_setup(opt):
    norm_method = Normalize([0, 0, 0] if opt.no_mean_norm else opt.mean, [1, 1, 1] if not opt.std_norm else opt.std)
    spatial_transform = Compose([Scale(opt.sample_size), CenterCrop(opt.sample_size), ToTensor(opt.norm_value), norm_method])
    temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform, ehw_cam=opt.ehw_cam, ehw_input=opt.ehw_input)
    val_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=16, shuffle=False, num_workers=opt.n_threads, pin_memory=True)
    val_logger = Logger(os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'action_prec1', 'quality_prec1'])
    return val_loader, val_logger, validation_data.class_to_idx

def get_test_setup(opt):
    norm_method = Normalize([0, 0, 0] if opt.no_mean_norm else opt.mean, [1, 1, 1] if not opt.std_norm else opt.std)
    spatial_transform = Compose([Scale(int(opt.sample_size / opt.scale_in_test)), CornerCrop(opt.sample_size, opt.crop_position_in_test), ToTensor(opt.norm_value), norm_method])
    temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
    target_transform = VideoID()
    test_data = get_test_set(opt, spatial_transform, temporal_transform, target_transform, ehw_cam=opt.ehw_cam, ehw_input=opt.ehw_input)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=16, shuffle=False, num_workers=opt.n_threads, pin_memory=True)
    return test_loader, test_data.class_names