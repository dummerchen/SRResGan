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
class BSD_DataSets(Dataset):
    def __init__(self,path,train_or_val='train',scale_factor=4):
        self.image_paths=glob(os.path.join(path,train_or_val,'*.jpg'))
        self.train_or_val=train_or_val
        self.scale_factor=scale_factor
        self.transforms={
            'train':
                transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(size=96),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]),
            'val':transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
        }

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, item):
        # h,w,c bgr
        image=cv2.imread(self.image_paths[item])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # c,h,w
        hr=self.transforms[self.train_or_val](image)

        c,h,w=hr.shape
        lr=transforms.Resize(size=[h//self.scale_factor,w//self.scale_factor])(hr)
        hr=transforms.CenterCrop(size=(h//self.scale_factor*self.scale_factor,w//self.scale_factor*self.scale_factor))(hr)
        return lr,hr


