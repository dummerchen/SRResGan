# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/4 12:48
import cv2
import torch
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from model.SRResNet import SRResNet
from model.models import SRResNet as sr

if __name__ == '__main__':
    scale_factor=4
    path='../datasets/bsds500/val/2018.jpg'
    image = cv2.imread(path)
    h,w,c=image.shape
    if not os.path.exists('./results'):
        os.mkdir('./results')
    bimage=cv2.resize(image,(w*4,h*4),interpolation=3)
    cv2.imwrite('./results/lr_b.png',bimage)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    device='cpu'

    # 19.79 0.752
    # weights_path='./weights/SRResNet_30.pth'
    # weights_path='./weights/SRGan_50.pth'
    weights_path= 'weights/checkpoint_srresnet_21.pth'
    # c,h,w
    transform = {
        'train':
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(size=96),
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
    model = SRResNet(in_channels=3, n_block=16, scale_factor=4, hidden_channels=64)
    # model = sr()
    # 用网上这个人的https://github.com/luoming1994/SRResNet训练的效果还没我的好 锐化太严重
    params = torch.load(weights_path, map_location=device)
    # model.load_state_dict(params['model'])
    model.load_state_dict(params['weights'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        res = model(lr)
        res=res.squeeze().numpy().transpose(1,2,0)
        lr=lr.squeeze().numpy().transpose(1,2,0)
        res=(res+1.)/2.
        # cv2.imshow('lr',lr)
        # cv2.imshow('res',res)
        cv2.imwrite('./results/gan_lr.png',cv2.cvtColor(lr*255,cv2.COLOR_RGB2BGR))
        cv2.imwrite('./results/gan_res.png',cv2.cvtColor(res*255,cv2.COLOR_RGB2BGR))
        cv2.waitKey()
        cv2.destroyAllWindows()