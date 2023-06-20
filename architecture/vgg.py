import torch.nn as nn

def VGG11(output_dim): return VGG_CIFAR([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], output_dim)
def VGG13(output_dim): return VGG_CIFAR([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], output_dim)
def VGG16(output_dim): return VGG_CIFAR([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], output_dim)
def VGG19(output_dim): return VGG_CIFAR([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], output_dim)

class VGG_CIFAR(nn.Module):
    def __init__(self, vgg_cfg, output_dim=10, bias=False, affine=False):
        super(VGG_CIFAR, self).__init__()
        self.bias = bias
        self.affine = affine
        self.features = self._make_layers(vgg_cfg)
        self.classifier = nn.Linear(512, output_dim, bias=self.bias)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
                           nn.BatchNorm2d(x, affine=self.affine),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
