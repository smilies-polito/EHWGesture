import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNeXt', 'resnext50', 'resnext101', 'resnext152']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400,
                 cardinality=32,
                 include_top = True):
        self.include_top = include_top
        self.cardinality = cardinality
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        #self.conv1 = nn.Conv3d(
        #    3,
        #    64,
        #    kernel_size=(3,7,7),
        #    stride=(1, 2, 2),
        #    padding=(1, 3, 3),
        #    bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        #last_duration = 1
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool3d(x, x.data.size()[-3:]).view(x.size(0), -1)
        return x

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # breakpoint()
        # x = self.avgpool(x)
        # # x = x.view(x.size(0), -1)
        x = F.avg_pool3d(x, x.data.size()[-3:]).view(x.size(0), -1)
        if self.include_top:
            x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")

class MultiInputResNext(nn.Module):
    def __init__(self, block, layers, sample_size, sample_duration, num_classes=400, input_heads=2, cardinality=32):
        super(MultiInputResNext, self).__init__()
        self.input_heads = input_heads
        
        # Create independent feature extractors for each input head
        self.feature_extractors = nn.ModuleList([
            ResNeXt(block, layers, sample_size, sample_duration, num_classes=num_classes, include_top=False)
            for _ in range(input_heads)
        ])
        
        # breakpoint()
        # Adjust the classifier to handle concatenated features
        feature_size = cardinality * 32 * block.expansion * input_heads
        self.fc = nn.Linear(feature_size, num_classes)
            
    def forward(self, inputs):
        assert len(inputs) == self.input_heads, "Expected {} input streams".format(self.input_heads)
        
        # breakpoint()
        features = [extractor(input) for extractor, input in zip(self.feature_extractors, inputs)]
        
        # breakpoint()
        # Concatenate features from all input heads
        x = torch.cat(features, dim=1)
        x = self.fc(x)
        return x


def make_multi_input(single_model, input_heads=2):
    """
    Convert a single-input ResNet into a multi-input one by duplicating its feature extractor.
    """
    multi_model = MultiInputResNext(
        block=type(single_model.layer4[0]),
        layers=[len(single_model.layer1), len(single_model.layer2), len(single_model.layer3), len(single_model.layer4)],
        sample_size=single_model.conv1.kernel_size[1],
        sample_duration=single_model.conv1.kernel_size[0],
        num_classes=single_model.fc.out_features,
        input_heads=input_heads
    )
    
    # Copy weights to each feature extractor
    for extractor in multi_model.feature_extractors:
        extractor.load_state_dict(single_model.state_dict())
    
    return multi_model
def generate_model(opt):
    model = resnext152(
        num_classes=opt.n_classes,
        sample_size=opt.sample_size,
        sample_duration=opt.sample_duration
    )

    if opt.width_mult != 1:
        print("Warning: width_mult != 1 not supported for ResNeXt, ignoring.")

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model)

    return model, model.parameters()
def resnext50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
