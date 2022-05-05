# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/2 14:52
import torch
from glob import glob
import torchvision
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
from glob import glob
import cv2
from torchvision.transforms import functional
class BSD_DataSets(Dataset):
    def __init__(self,path,train_or_val='train',scale_factor=4):
        self.image_paths=glob(os.path.join(path,train_or_val,'*.jpg'))
        self.train_or_val=train_or_val
        self.scale_factor=scale_factor


    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, item):
        # h,w,c bgr
        image=cv2.imread(self.image_paths[item])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # c,h,w
        if self.train_or_val=='val':
            c, h, w = image.shape
            self.hscale = h // self.scale_factor
            self.wscale = w // self.scale_factor
        else:
            self.hscale=96//self.scale_factor
            self.wscale=96//self.scale_factor
        self.transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(size=96),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'lr': transforms.Compose([
                transforms.Resize(size=[self.hscale, self.wscale],interpolation=functional.InterpolationMode.BICUBIC),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'hr': transforms.Compose([
                # -1~1
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
            # 验证集尽可能取最大为高分辨率
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(size=(self.hscale * self.scale_factor, self.wscale * self.scale_factor)),
                transforms.ToTensor(),
            ])
        }

        image=self.transforms[self.train_or_val](image)
        hr=self.transforms['hr'](image)
        lr=self.transforms['lr'](image)

        return lr,hr


