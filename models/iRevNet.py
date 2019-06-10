"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR, 2018

(c) Joern-Henrik Jacobsen, 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .model_utils import split, merge, injective_pad, psi

class SEBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(SEBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        x = merge(x[0], x[1])

        out = F.relu(self.bn1(x))

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        # print("w shale: ",w.shape)
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))

        w_sort,w1 = w.sort(1,descending=True)

        clip_val = w_sort[:,int(w_sort.shape[1]*0.8),:,:]
        clip_val = clip_val.unsqueeze(dim=-1)

        w_mask_gt = (w_sort>clip_val)
        w_relu = w_sort.mul(w_mask_gt.float())

        # Excitation
        out = x * w_relu

        x1, x2 = split(out)

        return (x1, x2)

class head_block(nn.Module):
    def __init__(self, nClasses, ds, channels):
        """ buid head block """
        super(head_block, self).__init__()
        self.nClasses = nClasses
        self.ds = ds
        self.bn = nn.BatchNorm2d(int(channels/2), momentum=0.9)
        self.linear = nn.Linear(int(channels/2), nClasses)
        
        print('')
        print('| Injective iRevNet |')
        print('')

    def forward(self, x):
        """ bijective or injective block forward """
        out = F.relu(self.bn(x))
        out = F.avg_pool2d(out, self.ds)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, split_ueq=False, dropout_rate=0.,
                 affineBN=True, mult=4):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.split_ueq = split_ueq
        self.pad = 2 * out_ch - in_ch
        self.stride = stride
        self.inj_pad = injective_pad(self.pad)
        self.psi = psi(stride)
        if self.pad != 0 and stride == 1:
            in_ch = out_ch * 2
            print('')
            print('| Injective iRevNet |')
            print('')
        layers = []

        if split_ueq:
            if not first:
                layers.append(nn.BatchNorm2d(in_ch//2, affine=affineBN))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_ch//2, int(out_ch//mult), kernel_size=3,
                        stride=stride, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(int(out_ch//mult), int(out_ch//mult),
                        kernel_size=3, padding=1, bias=False))
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(int(out_ch//mult), out_ch, kernel_size=3,
                        padding=1, bias=False))

        else:
            if not first:
                layers.append(nn.BatchNorm2d(in_ch//2, affine=affineBN))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_ch//2, int(out_ch//mult), kernel_size=3,
                        stride=stride, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(int(out_ch//mult), int(out_ch//mult),
                        kernel_size=3, padding=1, bias=False))
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(int(out_ch//mult), out_ch, kernel_size=3,
                        padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        if self.pad != 0 and self.stride == 1:
            x = merge(x[0], x[1])
            # print("x before inj_pad: ",x.shape)
            x = self.inj_pad.forward(x)

            x1, x2 = split(x)
            x = (x1, x2)
        
        x1 = x[0]
        x2 = x[1]

        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        if self.pad != 0 and self.stride == 1:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class iRevNet(nn.Module):
    def __init__(self, nBlocks, nStrides, nClasses, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4):
        super(iRevNet, self).__init__()
        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        print('')
        print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3 + 1))
        if not nChannels:
            nChannels = [self.in_ch//2, self.in_ch//2 * 4,
                         self.in_ch//2 * 4**2, self.in_ch//2 * 4**3]
        
        self.init_psi = psi(self.init_ds)

        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)
        
        self.bn1 = nn.BatchNorm2d(nChannels[-1]*2, momentum=0.9)
        self.linear = nn.Linear(nChannels[-1]*2, nClasses)

        self.se_block = SEBlock(32,32)

        # self.head_block0 = head_block(nClasses,self.ds,nChannels[-1]*2)
        # self.head_block1 = head_block(nClasses,self.ds,nChannels[-1]*2)
        

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, x):
        """ irevnet forward """
        n = self.in_ch//2
        if self.init_ds != 0:
            x = self.init_psi.forward(x)

        out = (x[:, :n, :, :], x[:, n:, :, :])

        idx = 0
        for block in self.stack:
            out = block.forward(out)
            if(idx==17):
                out = self.se_block(out)
                out_32x32 = merge(out[0], out[1])

            idx += 1

        out_bij = merge(out[0], out[1])

        out = F.relu(self.bn1(out_bij))
        out = F.avg_pool2d(out, self.ds)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out_32x32

    def inverse(self, out_bij):
        """ irevnet inverse """
        out = split(out_bij)
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)
        out = merge(out[0],out[1])
        if self.init_ds != 0:
            x = self.init_psi.inverse(out)
        else:
            x = out
        return x


if __name__ == '__main__':
    model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                    nChannels=None, nClasses=1000, init_ds=2,
                    dropout_rate=0., affineBN=True, in_shape=[3, 224, 224],
                    mult=4)
    y = model(Variable(torch.randn(1, 3, 224, 224)))
    print(y.size())
