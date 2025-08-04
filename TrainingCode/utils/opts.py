import argparse
import os
from utils.mean import get_mean, get_std

NETWORKS = {
    "phinet3d": "models.phinet3d_c3i",
    "resnet": "models.resnet",
    "resnet50": "models.resnet",
    "resnext": "models.resnext",
}

def import_network(network_name):
    if network_name in NETWORKS:
        module = __import__(NETWORKS[network_name], fromlist=["make_multi_input", "generate_model"])
        return module.make_multi_input, module.generate_model
    else:
        raise ValueError(f"Unknown network: {network_name}")
    
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ehw_cam', default='master', type=str)
    parser.add_argument('--ehw_input', default='rgb', type=str)
    
    parser.add_argument('--n_epochs', default=15, type=int, help='Number of total epochs to run')    
    parser.add_argument('--pretrain_epochs', default=1, type=int, help='Number of pretrain epochs to run')

    parser.add_argument('--root_path', default='~/', type=str, help='Root directory path of data')
    parser.add_argument('--video_path', default='~/EHWDataset/', type=str, help='Directory path of Videos')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--dataset', default='ehwgesture', type=str, help='Used dataset')
    parser.add_argument('--n_classes', default=11, type=int, help='Number of classes')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
    parser.add_argument('--time_downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
    parser.add_argument('--random_mask_fraction', default=0.0, type=float, help='Fraction of frames to randomly mask')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str, help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_steps', default=[10, 20, 40], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--ft_portion', default='complete', type=str, help='The portion of the model to apply fine tuning, either complete or last_layer')
    
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(no_val=False)
    parser.set_defaults(test=False)
    parser.add_argument('--test_subset', default='val', type=str, help='Used subset in test (val | test)')
    parser.add_argument('--scale_in_test', default=1.0, type=float, help='Spatial scale in test')
    parser.add_argument('--crop_position_in_test', default='c', type=str, help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument('--no_softmax_in_test', action='store_true', help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=10, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--no_hflip', action='store_true', help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--model', default='phinet3d', type=str, help='phinet3d')
    parser.add_argument('--groups', default=3, type=int, help='The number of groups at group convolutions at conv layers')
    parser.add_argument('--width_mult', default=1.0, type=float, help='The applied width multiplier to scale number of filters')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    opt = parser.parse_args()

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
    opt.scales = [opt.initial_scale]
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.width_mult) + 'x', str(opt.sample_duration)])
    opt.ehw_cam = opt.ehw_cam.split(',') if opt.ehw_cam else []
    opt.ehw_input = opt.ehw_input.split(',') if opt.ehw_input else []

    # breakpoint()
    return opt
