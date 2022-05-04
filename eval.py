# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/4 12:48
import cv2
import torch
from torchvision import transforms

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model.SRResNet import SRResNet

if __name__ == '__main__':
    scale_factor=4
    path='../datasets/bsds500/val/2018.jpg'
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    device='cpu'
    weights_path='./weights/SRResNet_1000.pth'
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
        ])
    }
    hr = transform['val'](image)

    c, h, w = hr.shape
    lr = transforms.Resize(size=[h // scale_factor, w // scale_factor])(hr)
    hr = transforms.CenterCrop(
        size=(h // scale_factor * scale_factor, w // scale_factor * scale_factor))(hr)

    lr, hr = lr.to(device), hr.to(device)
    lr=torch.unsqueeze(lr,dim=0)
    hr=torch.unsqueeze(hr,dim=0)

    model = SRResNet(in_channels=3, n_block=16, scale_factor=4, hidden_channels=64)

    # 用网上这个人的https://github.com/luoming1994/SRResNet训练的效果还没我的好 锐化太严重
    params = torch.load(weights_path, map_location=device)
    model.load_state_dict(params['weights'])
    model.to(device)
    model.eval()
    res = model(lr)

    with torch.no_grad():
        psnr = peak_signal_noise_ratio(hr.cpu().numpy(), res.cpu().numpy())
        ssim = structural_similarity(hr.cpu().numpy().squeeze().transpose(2, 1, 0),
                                     res.cpu().numpy().squeeze().transpose(2, 1, 0), multichannel=True)
        hr=hr.squeeze().numpy().transpose(1,2,0)
        res=res.squeeze().numpy().transpose(1,2,0)
        lr=lr.squeeze().numpy().transpose(1,2,0)

        cv2.imshow('hr',hr)
        cv2.imshow('res',res)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print(psnr,ssim)