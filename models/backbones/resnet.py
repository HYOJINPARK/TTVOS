import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os



__all__ = ['resnet18s16', 'resnet50s16', 'resnet101s16']#, 'hrnetv1w', 'hrnetv1wrx', 'hrnetv1wmv1']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetS16(ResNet):
    def __init__(self, finetune_layers, s16_feats, s8_feats, s4_feats, block, layers, num_classes=1000):
        super().__init__(block, layers, num_classes)
        self.finetune_layers = finetune_layers
        self.s16_feats = s16_feats
        self.s8_feats = s8_feats
        self.s4_feats = s4_feats

        self.layer4[0].downsample[0].stride = (1, 1)
        if block == BasicBlock :
            self.layer4[0].conv1.stride = (1, 1)
            self.BlockType= 'BasicBlock'
            for layer in self.layer4[1:]:
                layer.conv2.dilation = (2, 2)
                layer.conv2.padding = (2, 2)
        else:
            self.BlockType = 'BottleNeck'
            for layer in self.layer3[4:]:
                layer.conv2.dilation = (2, 2)
                layer.conv2.padding = (2, 2)
            del self.layer4
        del self.avgpool
        del self.fc
        # del self.layer4

        if finetune_layers[0]=='Not':
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
        if self.BlockType == 'BottleNeck':
            return((1024, 512, 256))
        else:
            return ((512,128,64))

    def get_features(self, x):
        feats = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feats['conv1'] = x
        x = self.maxpool(x)
        x = self.layer1(x)
        feats['layer1'] = x
        x = self.layer2(x)
        feats['layer2'] = x
        x = self.layer3(x)
        feats['layer3'] = x
        if self.BlockType != 'BottleNeck':
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

def resnet18s16(pretrained=False,  finetune_layers=(), s16_feats=('layer4',),
                s8_feats=('layer2',), s4_feats=('layer1',), nn_weights_path='./', **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print(s16_feats, s8_feats, s4_feats)
    model = ResNetS16(finetune_layers, s16_feats, s8_feats, s4_feats,BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained :
        if not os.path.isfile(nn_weights_path + "/resnet18-5c106cde.pth"):
            print('download ResNet 18 offical pretrained file')
            model_zoo.load_url(model_urls['resnet18'], model_dir=nn_weights_path)

        print("load pretrained ResNet18 model")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(nn_weights_path + "/resnet18-5c106cde.pth", map_location=torch.device(DEVICE)),
                              strict=False)
    else:
        print("Do not load pretrained model")
    return model



def resnet50s16(pretrained=False, finetune_layers=(), s16_feats=('layer4',),
                s8_feats=('layer2',), s4_feats=('layer1',), nn_weights_path='./', **kwargs):
    print(s16_feats, s8_feats, s4_feats)
    model = ResNetS16(finetune_layers, s16_feats, s8_feats, s4_feats, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained :
        if not os.path.isfile(nn_weights_path+"/resnet50-19c8e357.pth"):
            print('download ResNet 50 offical pretrained file')
            model_zoo.load_url(model_urls['resnet50'],model_dir=nn_weights_path)
        print("load pretrained ResNet50 model")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(nn_weights_path+"/resnet50-19c8e357.pth",map_location=torch.device(DEVICE)),
                              strict=False)
    else:
        print("Do not load pretrained model")
    return model


def resnet101s16(pretrained=False, finetune_layers=(), s16_feats=('layer4',),
                 s8_feats=('layer2',), s4_feats=('layer1',),nn_weights_path='./',  **kwargs):
    model = ResNetS16(finetune_layers, s16_feats, s8_feats, s4_feats, Bottleneck, [3, 4, 23, 3], **kwargs)
    file_name = model_urls['resnet101'].split('models/')[-1]
    if pretrained and os.path.isfile(nn_weights_path + file_name):
        if not os.path.isfile(nn_weights_path+file_name):
            print('download ResNet 101 offical pretrained file')
            model_zoo.load_url(model_urls['resnet101'],model_dir=nn_weights_path)

        print("load pretrained ResNet101 model")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(nn_weights_path + file_name,
                                         map_location=torch.device(DEVICE)), strict=False)
    else:
        print("Do not load pretrained model")
    return model
