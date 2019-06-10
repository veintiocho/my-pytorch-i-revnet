'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': ['M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': ['M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': ['M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name

        conv1_layers = []
        conv1_layers += [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                           nn.BatchNorm2d(64),
                           nn.ReLU(inplace=True)]
        self.conv1 = nn.Sequential(*conv1_layers)

        if(vgg_name != 'VGG11'):
            conv2_layers = []
            conv2_layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True)]
            self.conv2 = nn.Sequential(*conv2_layers)


        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out1 = self.conv1(x)
        if(self.vgg_name != 'VGG11'):
            out2 = self.conv2(out1)

        out = self.features(out2)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 64
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11',3)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
