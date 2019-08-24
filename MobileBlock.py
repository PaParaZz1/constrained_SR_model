import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., self.inplace) / 6.

# maybe these functional should be put on a utils file
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out =  F.relu6(x + 3., self.inplace) / 6.
        return out * x



class SEBlock(nn.Module):
    # squeeze and excite block
    ''' x-avgpool-Linear(down)--ReLU--Linear(up)--h_sigmoid-*->
        |___________________________________________________|
    '''
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)

        return  x * y.expand_as(x)


class MobileBlock(nn.Module):
    '''exp_size for 1x1 conv'''
    def __init__(self, nf, exp_size, kernel_size, stride, activation='relu'):
        super(MobileBlock, self).__init__()

        if activation == "relu":
            self.activation = nn.ReLU
        else:
            self.activation = h_swish
        self.conv = nn.Sequential(
            nn.Conv2d(nf, exp_size, kernel_size=1, stride=1, padding=0, bias = False)
            #, nn.BatchNorm2d(exp_size),
        )

        padding = (kernel_size - 1) // 2
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride, padding=padding, groups=exp_size)
            #, nn.BatChNorm2d(exp_size),
        )

        self.se_block = SEBlock(exp_size, reduction=4)
        
        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, nf, kernel_size=1, stride=1, padding=0)
            #,nn.BatchNorm2d(nf)
            , self.activation(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        out = self.depth_conv(out)
        out = self.se_block(out)
        out = self.point_conv(out)

        return x + out

if __name__ == '__main__':
    block = MobileBlock(64,64,  3, 1, activation='h-swish')
    x = torch.randn(4, 64, 400, 500)
    print(x.shape)
    print(block(x).shape)


