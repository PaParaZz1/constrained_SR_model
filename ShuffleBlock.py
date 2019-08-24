import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np

def conv3x3(in_channels, out_channels, stride, padding=1, groups=1):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                    kernel_size=3, stride=stride, padding=padding,
                    groups=groups,
                    bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                    kernel_size=1, stride=stride,padding=0,
                    bias=False)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish,self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x

class ShuffleBlock(nn.Module):
'''keep downsample None'''
    def __init__(self, inplanes, planes, stride=1,activation = 'relu', downsample=None):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample

        if not self.downsample: #---if not downsample, then channel split, so the channel become half
            inplanes = inplanes // 2
            planes = planes // 2
 
        self.conv1x1_1 = conv1x1(in_channels=inplanes, out_channels=planes)
        #self.conv1x1_1_bn = nn.BatchNorm2d(planes)

        self.dwconv3x3 = conv3x3(in_channels=planes, out_channels=planes, stride=stride, groups=planes)
        #self.dwconv3x3_bn= nn.BatchNorm2d(planes)

        self.conv1x1_2 = conv1x1(in_channels=planes, out_channels=planes)
        #self.conv1x1_2_bn = nn.BatchNorm2d(planes)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation =  h_swish(inplace=True)

    def _channel_split(self, features, ratio=0.5):
        """
        ratio: c'/c, default value is 0.5
        """ 
        size = features.size()[1]
        split_idx = int(size * ratio)
        return features[:,:split_idx], features[:,split_idx:]

    def _channel_shuffle(self, features, g=2):
        channels = features.size()[1] 
        index = torch.from_numpy(np.asarray([i for i in range(channels)]))
        index = index.view(-1, g).t().contiguous()
        index = index.view(-1).cuda()
        features = features[:, index]
        return features

    def forward(self, x):
        if  self.downsample:
            #x1 = x.clone() #----deep copy x, so where x2 is modified, x1 not be affected
            x1 = x
            x2 = x
        else:
            x1, x2 = self._channel_split(x)

        #----right branch----- 
        x2 = self.conv1x1_1(x2)
        #x2 = self.conv1x1_1_bn(x2)
        x2 = self.activation(x2)
         
        x2 = self.dwconv3x3(x2)
        #x2 = self.dwconv3x3_bn(x2)
    
        x2 = self.conv1x1_2(x2)
        #x2 = self.conv1x1_2_bn(x2)
        x2 = self.activation(x2)

        #---left branch-------
        if self.downsample:
            x1 = self.downsample(x1)

        x = torch.cat([x1, x2], 1)
        x = self._channel_shuffle(x)
        return x



if __name__ == "__main__":
    x = torch.randn(4, 64, 400, 500)
    block = ShuffleBlock(64, 64, 'h-swish')
    print(x.shape)
    print(block(x).shape)
