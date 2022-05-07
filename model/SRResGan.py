# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/4 11:30
import torch
import torchvision
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


class TruncatedVGG19(nn.Module):
    """
    truncated VGG19网络，用于计算VGG特征空间的MSE损失
    """

    def __init__(self, i, j):
        """
        :参数 i: 第 i 个池化层
        :参数 j: 第 j 个卷积层
        """
        super(TruncatedVGG19, self).__init__()

        # 加载预训练的VGG模型
        vgg19 = torchvision.models.vgg19(
            pretrained=True)  # C:\Users\Administrator/.cache\torch\checkpoints\vgg19-dcbb9e9d.pth

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # 迭代搜索
        for layer in vgg19.features.children():
            truncate_at += 1

            # 统计
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # 截断位置在第(i-1)个池化层之后（第 i 个池化层之前）的第 j 个卷积层
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # 检查是否满足条件
        assert maxpool_counter == i - 1 and conv_counter == j, "当前 i=%d 、 j=%d 不满足 VGG19 模型结构" % (
            i, j)

        # 截取网络
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        前向传播
        参数 input: 高清原始图或超分重建图，张量表示，大小为 (N, 3, w * scaling factor, h * scaling factor)
        返回: VGG19特征图，张量表示，大小为 (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output

if __name__ == '__main__':
    input=torch.randn((4,3,96,96))
    model=Discriminator(in_channels=3)
    out=model(input)