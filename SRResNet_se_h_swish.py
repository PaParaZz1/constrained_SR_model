import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import argparse
import cv2
from srresnet import MSRResNet


def make_model(args):
    return MSRResNet_v1()


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

# ----
# Se Layer: refer to https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, activation='relu'):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if activation == 'relu':
            self.activation = nn.ReLU
        else:
            self.activation = h_swish

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            self.activation(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ----
# h-swish: refer to https://github.com/leaderj1001/MobileNetV3-Pytorch/blob/master/model.py
# ----
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64, activation='relu'):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'h-swish':
            self.activation = h_swish(inplace=True)
        else:
            print('activation functino must be relu or h-swish')

        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def naive_bn(self, x):
        a = self.bn.weight.view(1, -1, 1, 1)
        return x.mul_(a)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.naive_bn(x)
        x = self.activation(x)
        out = self.conv2(x)
        return identity + out


class SE_ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+---SE-->
     |________________|
    '''

    def __init__(self, nf=64, activation='relu', use_bn=False):
        super(SE_ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'h-swish':
            self.activation = h_swish(inplace=True)
        else:
            print('activation functino must be relu or h-swish')

        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.se = SELayer(nf, reduction=16, activation=activation)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(nf)

    def naive_bn(self, x):
        a = self.bn.weight.view(1, -1, 1, 1)
        return x.mul_(a)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        # defalut: out = F.relu(self.conv1(x), inplace=True)
        if self.use_bn:
            out = self.naive_bn(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.se(out)  # se layers
        return identity + out


class MSRResNet_v1(nn.Module):
    ''' modified SRResNet'''
    ''' benchmark: nf = 64, nb = 16, activation = relu , mode  = benchmark'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=12, upscale=4, activation='h-swish', mode='se'):
        super(MSRResNet_v1, self).__init__()

        if activation not in ['relu', 'h-swish']:
            print('activation must be \'relu\' or \'h-swish\'')

        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        if mode == 'se':
            block = functools.partial(
                SE_ResidualBlock_noBN, nf=nf, activation=activation)
        elif mode == 'benchmark':
            block = functools.partial(
                ResidualBlock_noBN, nf=nf, activation=activation)
        else:
            print('mode must be \'benchmark\' or \'se\'')

        self.recon_trunk = make_layer(block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function

        if activation == 'relu':
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.activation = h_swish(inplace=True)

        # initialization
        initialize_weights([self.conv_first, self.upconv1,
                            self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.activation(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.activation(self.pixel_shuffle(self.upconv1(out)))
            out = self.activation(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.activation(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.activation(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale,
                             mode='bilinear', align_corners=False)
        out += base
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', default='relu',
                        type=str, help='activation function type')
    parser.add_argument('--mode', default='benchmark',
                        type=str, help='use SE or not')

    args = parser.parse_args()
    print(args)
    net = MSRResNet_v1(activation=args.activation, mode=args.mode)
    net_bm = MSRResNet()
    print(net)

    number_parameters = sum(map(lambda x: x.numel(), net.parameters()))
    number_parameters_bm = sum(map(lambda x: x.numel(), net_bm.parameters()))

    print("number_parameters:", number_parameters)
    print("number_parameters_bm:", number_parameters_bm)

    img = cv2.imread('./0001x4.png')
    img = torch.Tensor(img)
    # print(img.shape)
    #w, h, c = img.shape
    #img = img.reshape(1, c, w, h)
    # print(net(img).shape)
