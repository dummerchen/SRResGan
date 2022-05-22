# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/2 14:52
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from glob import glob
import cv2
from torchvision.transforms import functional
from torch.utils.data.dataloader import default_collate
import logging

def collate(batch):
    if isinstance(batch, list):
        batch = [(lr, hr) for (lr, hr) in batch if lr is not None]
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)

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
        h, w, c = image.shape
        if h<96 or w<96:
            return None,None
        if self.train_or_val=='val':
            self.hscale = h // self.scale_factor
            self.wscale = w // self.scale_factor
        else:
            self.hscale=96//self.scale_factor
            self.wscale=96//self.scale_factor
        self.transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(size=96),
                transforms.ToTensor(),
            ]),
            'lr': transforms.Compose([
                transforms.Resize(size=[self.hscale, self.wscale],interpolation=functional.InterpolationMode.BICUBIC),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # 验证集尽可能取最大为高分辨率
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(size=(self.hscale * self.scale_factor, self.wscale * self.scale_factor)),
                transforms.ToTensor(),
            ])
        }

        image=self.transforms[self.train_or_val](image)
        hr=image
        lr=self.transforms['lr'](image)

        return lr,hr


class Logger():
    def __init__(self,file_path,level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=level)

        handler = logging.FileHandler(file_path)
        handler.setLevel(level)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    def info(self,info):
        return self.logger.info(info)

    def debug(self,info):
        return self.logger.info(info)

    def warning(self,info):
        return self.logger.warning(info)

    def error(self,info):
        return self.logger.error(info)

