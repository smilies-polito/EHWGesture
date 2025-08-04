from datasets.ehwgesture import EHWGestureDataset

def get_training_set(opt, spatial_transform, temporal_transform, target_transform, ehw_cam=['master', 'sub2'], ehw_input=['depth', 'rgb']):
    # breakpoint()
    training_data = EHWGestureDataset(
        opt.video_path,
        'train',
        input_type=ehw_input,
        camera=ehw_cam,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration,
        time_downsample=opt.time_downsample,
        random_mask_fraction=opt.random_mask_fraction)
    return training_data

def get_validation_set(opt, spatial_transform, temporal_transform, target_transform, ehw_cam=['master', 'sub2'], ehw_input=['depth', 'rgb']):
    validation_data = EHWGestureDataset(
        opt.video_path,
        'val',
        input_type=ehw_input,
        camera=ehw_cam,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration,
        time_downsample=opt.time_downsample,
        random_mask_fraction=opt.random_mask_fraction)
    return validation_data

def get_test_set(opt, spatial_transform, temporal_transform, target_transform, ehw_cam=['master', 'sub2'], ehw_input=['depth', 'rgb']):
    test_data = EHWGestureDataset(
        opt.video_path,
        opt.test_subset,
        input_type=ehw_input,
        camera=ehw_cam,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration,
        time_downsample=opt.time_downsample,
        random_mask_fraction=opt.random_mask_fraction)
    return test_data
