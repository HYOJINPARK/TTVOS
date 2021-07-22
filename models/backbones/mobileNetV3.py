import os
import logging
logger = logging.getLogger('global')
import torch
from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence

from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenetv2 import _make_divisible, ConvBNActivation


__all__ = ["mobileNetV3Smalls16", "mobileNetV3Larges16"]


model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):

    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)




class MobileNetV3Seg(MobileNetV3):

    def __init__(self,finetune_layers, pretrained, arch, nn_weights_path, s16_feats, s8_feats, s4_feats,
                 inverted_residual_setting,last_channel,
            num_classes = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None):
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__(inverted_residual_setting, last_channel, num_classes, block, norm_layer)


        if pretrained:

            if model_urls.get(arch, None) is None:
                raise ValueError("No checkpoint is available for model type {}".format(arch))
            url_name = model_urls[arch].split('models/')[-1]
            if not os.path.isfile(os.path.join(nn_weights_path, url_name)):
                state_dict = load_state_dict_from_url(model_urls[arch], nn_weights_path,
                                                      progress=False)
                print("done download model")
            else:
                state_dict = torch.load(os.path.join(nn_weights_path ,url_name))
                print("load pretrained model : " + url_name)
            self.load_state_dict(state_dict, strict=False)
            print("done initialized with pretrained model")
        else:
            print("train from scratch")


        self.finetune_layers = finetune_layers
        self.s16_feats = s16_feats
        self.s8_feats = s8_feats
        self.s4_feats = s4_feats
        self.arch =arch
        if arch == "mobilenet_v3_small":
            self.layer0 = self.features[0]
            self.layer1 = self.features[1]
            self.layer2 = self.features[2:4]
            self.layer3 = self.features[4:9]
            self.layer4 = self.features[9:]

        else :
            self.layer0 = self.features[0:2]
            self.layer1 = self.features[2:4]
            self.layer2 = self.features[4:7]
            self.layer3 = self.features[7:13]
            self.layer4 = self.features[13:]


        del self.avgpool
        del self.features
        del self.classifier

        if finetune_layers[0] == 'Not':
            print("Train every layers")
        else:
            self.requires_grad = False
            for param in self.parameters():
                param.requires_grad = False
            for module_name in finetune_layers:
                print(module_name + " trained ")

                getattr(self, module_name).train(True)
                getattr(self, module_name).requires_grad = True
                for param in getattr(self, module_name).parameters():
                    param.requires_grad = True

    def get_return_values(self, feats):
        return {'s16': torch.cat([feats[name] for name in self.s16_feats], dim=-1),
                's8': torch.cat([feats[name] for name in self.s8_feats], dim=-1),
                's4': torch.cat([feats[name] for name in self.s4_feats], dim=-1)}

    def get_feature_count(self):
        if self.arch =='mobilenet_v3_small':
            return ((576,24,16))
        else: return((960,40,24))


    def get_features(self, x):
        feats = {}
        x = self.layer0(x)
        x = self.layer1(x)
        feats['layer1'] = x
        x = self.layer2(x)
        feats['layer2'] = x
        x = self.layer3(x)
        feats['layer3'] = x
        x = self.layer4(x)
        feats['layer4'] = x
        return self.get_return_values(feats)

    def train(self, mode):
        if self.finetune_layers[0] == "Not":
            # BN freeze
            for name, module in self.named_children():
                module.train(mode)
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        else:
            for name, module in self.named_children():
                if name in self.finetune_layers:
                    module.train(mode)
                else:
                    module.train(False)

    def eval(self):
        self.train(False)


def _mobilenet_v3_conf(arch: str, params: Dict[str, Any]):
    # non-public config parameters
    reduce_divider = 2 if params.pop('_reduced_tail', False) else 1
    dilation = 2
    width_mult = params.pop('_width_mult', 1.0)

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1), #1
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # 2 C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1), #3
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # 4 C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1), # 5
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1), # 6
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # 7 C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1), #8
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1), #9
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),#10
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1), # 11
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1), # 12
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 1, dilation),  # 13 C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation), #14
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation), #15
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # 16 C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1 1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C22
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1), #3
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3 4
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1), #5
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1), #6
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1), #7
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1), #8
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 1, dilation),  # C4 #9
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),#10
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),#11
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting, last_channel


def _mobilenet_v3_model(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)



def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)


def mobileNetV3Smalls16(pretrained: bool = False, finetune_layers=('stage4',), s16_feats=('layer4',), s8_feats=('layer2',),
                s4_feats=('layer1',), nn_weights_path='./', **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    model = MobileNetV3Seg(finetune_layers, pretrained, arch, nn_weights_path, s16_feats, s8_feats,
                           s4_feats,inverted_residual_setting,last_channel, **kwargs)
    return model


def mobileNetV3Larges16(pretrained: bool = False, finetune_layers=('stage4',), s16_feats=('layer4',), s8_feats=('layer2',),
                s4_feats=('layer1',), nn_weights_path='./', **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    model = MobileNetV3Seg(finetune_layers, pretrained, arch, nn_weights_path, s16_feats, s8_feats,
                           s4_feats,inverted_residual_setting,last_channel,**kwargs)
    return model
#
#
if __name__ == '__main__':
    import torchvision.models as models
    param={'_dilated':4}
    mobilenet_v3_small = models.mobilenet_v3_large(pretrained=False, progress=False, _dilated=4)
    arch = "mobilenet_v3_large"
#     A=dict()
#     # inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
#     model= _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, False, False)
#     print(model)
