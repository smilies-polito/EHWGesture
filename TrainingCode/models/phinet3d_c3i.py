import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Phinet3D(nn.Module):
    def __init__(self, num_classes=1000, sample_size=224, width_mult=1., input_heads=1):
        super(Phinet3D, self).__init__()
        block = InvertedResidual
        self.num_classes = num_classes
        self.sample_size = sample_size
        self.width_mult = width_mult
        self.num_inputs = input_heads
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            [1,  16, 1, (1,1,1)],
            [6,  24, 2, (2,2,2)],
            [6,  32, 3, (2,2,2)],
            [6,  64, 4, (2,2,2)],
            [6,  96, 3, (1,1,1)],
            [6, 160, 3, (2,2,2)],
            [6, 320, 1, (1,1,1)],
        ]

        assert sample_size % 16 == 0.
        input_channels = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.input_heads = input_heads
        
        self.feature_extractors = nn.ModuleList()
        for _ in range(input_heads):
            input_channel = input_channels
            features = [conv_bn(3, input_channel, (1,2,2))]
            for t, c, n, s in interverted_residual_setting:
                output_channel = int(c * width_mult)
                for i in range(n):
                    stride = s if i == 0 else (1,1,1)
                    # print(f"Block {i}:  output_channel:{output_channel}, input_channel:{input_channel}, stride:{stride}")
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
            features.append(conv_1x1x1_bn(input_channel, self.last_channel))
            self.feature_extractors.append(nn.Sequential(*features))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel * input_heads, num_classes),
        )
        
        self._initialize_weights()

    def forward(self, x_list):
        if isinstance(x_list, list):
            # print("input shape: ", x_list[0].shape)
            features = [head(x) for head, x in zip(self.feature_extractors, x_list)]
            features = [F.avg_pool3d(f, f.data.size()[-3:]).view(f.size(0), -1) for f in features]
            x = torch.cat(features, dim=1)
        else:
            # print("input shape: ", x_list.shape)
            x = self.feature_extractors[0](x_list)
            x = F.avg_pool3d(x, x.data.size()[-3:]).view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        x = self.feature_extractors[0](x)
        x = F.avg_pool3d(x, x.data.size()[-3:]).view(x.size(0), -1)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

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

def get_model(**kwargs):
    model = Phinet3D(**kwargs)
    return model

def get_single_input_model(**kwargs):
    return Phinet3D(input_heads=1, **kwargs)

def make_multi_input(single_input_model, num_inputs):
    multi_input_model = Phinet3D(num_classes=single_input_model.classifier[1].out_features, 
                                 sample_size=single_input_model.sample_size, width_mult=single_input_model.width_mult, input_heads=num_inputs)
    
    for i in range(num_inputs):
        multi_input_model.feature_extractors[i].load_state_dict(single_input_model.feature_extractors[0].state_dict())

    print("model transferred sucessfully")
    
    return multi_input_model

def generate_model(opt):
    model = get_single_input_model(
        num_classes=opt.n_classes,
        sample_size=opt.sample_size,
        width_mult=opt.width_mult,
    )

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model)

    return model, model.parameters()

if __name__ == "__main__":
    single_model = get_single_input_model(num_classes=600, sample_size=112, width_mult=1.)
    multi_model = make_multi_input(single_model, num_inputs=4)
    print(multi_model)


