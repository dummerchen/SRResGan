# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/4 12:48

import builtins
if 'builtins' not in dir() or not hasattr(builtins, 'profile'):
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner
    builtins.__dict__['profile'] = profile
import cv2
import torch
from torchvision import transforms
import os
from model.SRResNet import SRResNet
import time
if __name__ == '__main__':


    scale_factor=4
    # path='../datasets/bsds500/val/238025.jpg'
    path='./results/108036_1.jpg'
    t=time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./results/{}'.format(t)):
        os.mkdir('./results/{}'.format(t))

    image = cv2.imread(path)
    h,w,c=image.shape
    cv2.imwrite('./results/{}/hr.png'.format(t), image)
    bimage=cv2.resize(image,(w*4,h*4),interpolation=3)
    cv2.imwrite('./results/{}/lr_b.png'.format(t),bimage)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    device='cpu'

    # 19.79 0.752
    weights_path=['./weights/SRGan_sq_g_250.pth','./weights/SRResNet_sq_90.pth']
    # weights_path= 'weights/checkpoint_srresnet_21.pth'
    # c,h,w
    transform = {
        'lr':
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    lr = transform['val'](image)
    lr= lr.to(device)
    lr=torch.unsqueeze(lr,dim=0)

    for i,weight in enumerate(weights_path):
        model = SRResNet(in_channels=3, n_block=16, scale_factor=4, hidden_channels=64)
        model.to(device)
        # 用网上这个人的https://github.com/luoming1994/SRResNet训练的效果还没我的好 锐化太严重
        params = torch.load(weight, map_location=device)
        # model.load_state_dict(params['model'])
        model.load_state_dict(params['weights'])
        model.eval()
        with torch.no_grad():
            res = model(lr)
            res=res.squeeze().numpy().transpose(1,2,0)
            res=(res+1.)/2.
            cv2.imwrite('./results/{}/res_{}.png'.format(t,i),cv2.cvtColor(res*255,cv2.COLOR_RGB2BGR))
