# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/2 14:40
import time

import torch
from torch.utils import tensorboard
import os
import argparse
from tqdm import tqdm
from model.SRResNet import SRResNet
from bsd_dataset import BSD_DataSets
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main(opts):
    start_epoch=0
    device='cpu' if not torch.cuda.is_available() else 'cuda:0'
    print('using device:{}'.format(device))
    if not os.path.exists(opts.logs_dir):
        os.mkdir(opts.logs_dir)
    if not os.path.exists(opts.data_path_root):
        print('data not found！！！')
        return

    writer=tensorboard.SummaryWriter(log_dir=os.path.join(opts.logs_dir,time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))))
    if opts.workers==None:
        workers=min([os.cpu_count(), opts.batchsize if opts.batchsize > 1 else 0, 8])
    else:
        workers=opts.workers
    train_data=BSD_DataSets(opts.data_path_root,'train')
    val_data=BSD_DataSets(opts.data_path_root,'val')

    train_dataloader=DataLoader(train_data,shuffle=True,batch_size=opts.batchsize,num_workers=workers)
    val_dataloader=DataLoader(val_data,shuffle=False,batch_size=1,num_workers=workers)

    model=SRResNet(in_channels=3,n_block=16,scale_factor=4,hidden_channels=64)

    loss_func=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(lr=opts.learning_rate,params=model.parameters(),betas=(0.006,0))

    if  opts.weights!=None and os.path.exists(opts.weights):
        # 加载模型
        params=torch.load(opts.weights,map_location=device)
        optimizer=optimizer.load_state_dict(params['optimzer'])
        model.load_state_dict(params['weights'])
        start_epoch=params['epoch']

    for epoch in range(start_epoch,opts.epoch):
        model.train()
        train_bar=tqdm(train_dataloader)
        train_mean_loss=0
        train_mean_psnr=0
        for lr,hr in train_bar:
            optimizer.zero_grad()
            lr,hr=lr.to(device),hr.to(device)
            res=model(lr)
            loss=loss_func(res,hr)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                psnr = peak_signal_noise_ratio(hr.numpy(), res.numpy())
                train_mean_psnr+=psnr
                train_mean_loss+=loss.item()

                train_bar.set_description(desc='Epoch[{}/{}] Loss:{} PSNR:{}'.format(epoch,opts.epoch,loss.item(),psnr))

        train_mean_loss=train_mean_loss/len(train_bar)
        train_mean_psnr=train_mean_psnr/len(train_bar)
        writer.add_scalar('train_loss',train_mean_loss,epoch)
        writer.add_scalar('train_psnr',train_mean_psnr,epoch)

        if epoch%1==0:
            model.eval()
            val_bar=tqdm(val_dataloader)
            val_mean_loss = 0
            val_mean_psnr=0
            val_mean_ssim=0
            with torch.no_grad():
                for lr,hr in val_bar:
                    lr, hr = lr.to(device), hr.to(device)
                    res=model(lr)
                    loss=loss_func(res,hr)

                    val_mean_loss+=loss.item()
                    psnr=peak_signal_noise_ratio(hr.numpy(),res.numpy())
                    ssim=structural_similarity(hr.numpy().squeeze().transpose(2,1,0),res.numpy().squeeze().transpose(2,1,0),multichannel=True)

                    val_mean_psnr +=psnr
                    val_mean_ssim +=ssim
                    val_bar.set_description(desc='Epoch [{}/{}] Loss:{} PSNR:{} SSIM:{}'.format(epoch,opts.epoch,loss.item(),psnr,ssim))
                    # 每次只输出最后一组的验证结果
                    writer.add_images('SRResNet/epoch_' + str(epoch) + '_lr',
                                      lr, epoch)
                    writer.add_images('SRResNet/epoch_' + str(epoch) + '_res',
                                      res, epoch)
                    writer.add_images('SRResNet/epoch_' + str(epoch) + '_hr',
                                      hr, epoch)
                val_mean_loss/=len(val_bar)
                val_mean_psnr/=len(val_bar)
                val_mean_ssim/=len(val_bar)
                writer.add_scalar('val_mean_loss',val_mean_loss,epoch)
                writer.add_scalar('val_mean_psnr',val_mean_psnr,epoch)
                writer.add_scalar('val_mean_ssim',val_mean_ssim,epoch)
            d={
                'weights':model.state_dict(),
                'epoch':epoch,
                'optimizer':optimizer.state_dict(),

               }
            torch.save(d,'./weights/SRResNet_{}.pth'.format(epoch))
    writer.close()

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--data_path_root','-dpr',default='../datasets/bsds500',type=str)
    args.add_argument('--batchsize','-bs',default=2,type=int)
    args.add_argument('--weights','-w',default=None,type=str)
    args.add_argument('--logs_dir','-ld',default='./logs',type=str)
    args.add_argument('--scale_factor','-sf',default=4,type=int)
    args.add_argument('--learning_rate','-lr',default=0.0001,type=float)
    args.add_argument('--epoch','-e',default=100,type=int)
    args.add_argument('--workers','-wks',default=None,type=int)

    opts=args.parse_args()
    main(opts)