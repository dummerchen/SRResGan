# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/4 11:30
import torch
from torch import nn
class Discriminator(nn.Module):
    def __init__(self,in_channels):
        super(Discriminator, self).__init__()
        self.conv1=Block(in_channels=in_channels,kernel_size=3,hidden=64,stride=1,use_bn=False)

        self.blocks=nn.Sequential(*[
            Block(in_channels=64,hidden=64),
            Block(in_channels=64,hidden=128,stride=1),
            Block(in_channels=128,hidden=128,stride=2),
            Block(in_channels=128,hidden=256,stride=1),
            Block(in_channels=256,hidden=256,stride=2),
            Block(in_channels=256,hidden=512,stride=1),
            Block(in_channels=512,hidden=512,stride=2),
        ])
        # 这注定了d只能用于训练，因为输入维度不能改变
        self.fc1=nn.Linear(512*6*6,1024)
        self.ac=nn.LeakyReLU(0.2,inplace=True)
        self.fc2=nn.Linear(1024,1)
        self.sigmod=nn.Sigmoid()
    def forward(self,x):
        b,c,w,h=x.shape
        out1=self.conv1(x)
        out2=self.blocks(out1)
        out2=out2.view(b,-1)
        out3=self.fc1(out2)
        out4=self.ac(out3)
        out5=self.fc2(out4)
        out6=self.sigmod(out5)
        return out6

class Block(nn.Module):
    def __init__(self,in_channels,hidden=64,kernel_size=3,stride=2,use_bn=True,padding=1):
        super(Block, self).__init__()
        self.use_bn=use_bn
        self.conv=nn.Conv2d(in_channels,hidden,kernel_size=kernel_size,stride=stride,padding=padding,)
        self.bn=nn.BatchNorm2d(hidden)
        self.ac=nn.LeakyReLU(0.2,inplace=True)

    def forward(self,x):
        out1=self.conv(x)
        if self.use_bn==True:
            out=self.bn(out1)
        out=self.ac(out1)
        return out

if __name__ == '__main__':
    input=torch.randn((4,3,96,96))
    model=Discriminator(in_channels=3)
    out=model(input)