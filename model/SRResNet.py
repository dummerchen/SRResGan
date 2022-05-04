# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/2 14:40
import math

import torch
from torch import nn
class SRResNet(nn.Module):
    def __init__(self,in_channels,n_block,scale_factor,hidden_channels):
        super(SRResNet, self).__init__()
        self.num_blocks=n_block
        self.conv1=nn.Conv2d(in_channels,hidden_channels,kernel_size=(9,9),stride=(1,1),padding=4)
        self.ac1=nn.PReLU(num_parameters=1,init=0.2)
        self.blocks=nn.Sequential(*[Block(in_channels=hidden_channels,hidden=hidden_channels) for i in range(self.num_blocks)])

        self.conv2=nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm2d(hidden_channels)
        self.pixel_shuffle=nn.Sequential(*[SubPixelShuffleConvBlock(in_channels=hidden_channels) for i in range(scale_factor//2)])
        self.conv3=nn.Conv2d(hidden_channels,3,kernel_size=(9,9),stride=1,padding=4)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                if m.bias!=None:
                    m.bias.data.zero_()
    def forward(self,x):
        out1=self.conv1(x) # B,64,w,h
        out1=self.ac1(out1)
        out2=self.blocks(out1) # B,64,w,h
        out3=self.conv2(out2) # B,64,w,h
        out3=self.bn(out3)
        out4=out1+out3
        out5=self.pixel_shuffle(out4) # B,64,w*4,h*4
        out6=self.conv3(out5) # B,3,w*4,h*4
        return out6

class Block(nn.Module):
    def __init__(self,in_channels,kernal_size=3,hidden=64,stride=1):
        super(Block, self).__init__()
        self.conv1=nn.Conv2d(in_channels,hidden,kernel_size=kernal_size,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(hidden)
        self.ac1=nn.PReLU(num_parameters=1,init=0.2)
        self.conv2=nn.Conv2d(hidden,hidden,kernal_size,stride,padding=1)
        self.bn2=nn.BatchNorm2d(hidden)

    def forward(self,x):
        out1=self.conv1(x)
        out2=self.bn1(out1)
        out3=self.ac1(out2)
        out4=self.conv2(out3)
        out5=self.bn2(out4)
        out6=x+out5
        return out6

class SubPixelShuffleConvBlock(nn.Module):
    def __init__(self,in_channels,kernel_size=3,stride=1,hidden=256):
        super(SubPixelShuffleConvBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels,hidden,kernel_size,stride,padding=1)
        self.ps=nn.PixelShuffle(2)
        self.ac=nn.PReLU(num_parameters=1,init=0.2)
    def forward(self,x):
        out1=self.conv1(x) # B,256,w,h
        out2=self.ps(out1) # B,64,w*2,h*2
        out3=self.ac(out2)
        return out3

if __name__ == '__main__':
    model=SRResNet(in_channels=3,n_block=5,scale_factor=4,hidden_channels=64)
    input=torch.randn((4,3,128,128))
    out=model(input)
    print(out.shape)