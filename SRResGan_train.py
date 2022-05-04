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
from model.SRResNet import SRResNet as G
from model.SRResGan import Discriminator as D
from bsd_dataset import BSD_DataSets
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.backends.cudnn as cudnn

def main(opts):
    start_epoch=-1
    device='cpu' if not torch.cuda.is_available() else 'cuda:0'
    print('using device:{}'.format(device))
    if not os.path.exists(opts.logs_dir):
        os.mkdir(opts.logs_dir)
    if not os.path.exists(opts.data_path_root):
        print('data not found！！！')
        return
    cudnn.benchmark=True
    torch.manual_seed(opts.seed)

    writer=tensorboard.SummaryWriter(log_dir=os.path.join(opts.logs_dir,'Gan_'+time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))))

    if opts.workers==None:
        workers=min([os.cpu_count(), opts.batchsize if opts.batchsize > 1 else 0, 8])
    else:
        workers=opts.workers
    print('use workers:{}'.format(workers))

    train_data=BSD_DataSets(opts.data_path_root,'train')
    val_data=BSD_DataSets(opts.data_path_root,'val')

    train_dataloader=DataLoader(train_data,shuffle=True,batch_size=opts.batchsize,num_workers=workers,pin_memory=True)
    val_dataloader=DataLoader(val_data,shuffle=False,batch_size=1,num_workers=workers,pin_memory=True)

    g=G(in_channels=3,n_block=16,scale_factor=4,hidden_channels=64)
    g.to(device)
    d=D(in_channels=3)
    d.to(device)

    g_loss_func=torch.nn.MSELoss(size_average=False)
    d_loss_func=torch.nn.BCEWithLogitsLoss()

    g_optimizer=torch.optim.Adam(lr=opts.learning_rate,params=g.parameters(),betas=(0.9,0))
    d_optimizer=torch.optim.Adam(lr=opts.learning_rate,params=d.parameters(),betas=(0.9,0))
    g_lr_schedule=torch.optim.lr_scheduler.StepLR(g_optimizer,step_size=60,gamma=0.3)
    d_lr_schedule=torch.optim.lr_scheduler.StepLR(d_optimizer,step_size=60,gamma=0.3)

    if  opts.weights!=None and os.path.exists(opts.weights):
        # 加载模型
        params=torch.load(opts.weights,map_location=device)
        g_optimizer.load_state_dict(params['g_optimizer'])
        d_optimizer.load_state_dict(params['d_optimizer'])
        g.load_state_dict(params['g_weights'])
        d.load_state_dict(params['d_weights'])
        start_epoch=params['epoch']
        try:
            g_lr_schedule.load_state_dict(params['g_schedule'])
            d_lr_schedule.load_state_dict(params['d_schedule'])
        except:
            pass
    for epoch in range(start_epoch+1,opts.epoch+1):
        g.train()
        d.train()
        train_bar=tqdm(train_dataloader)
        train_mean_loss=0
        train_mean_psnr=0
        for lr,hr in train_bar:
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            lr,hr=lr.to(device),hr.to(device)
            res=g(lr)
            gloss=g_loss_func(res,hr)
            dloss=d_loss_func(res,hr)

            gloss.backward()
            g_optimizer.step()
            d_optimizer.step()
            with torch.no_grad():
                psnr = peak_signal_noise_ratio(hr.cpu().numpy(), res.cpu().numpy())
                train_mean_psnr+=psnr
                train_mean_loss+=loss.item()

                train_bar.set_description(desc='Train Epoch[{}/{}] lr:{} Loss:{} PSNR:{}'.format(epoch,opts.epoch,optimizer.param_groups[0]['lr'],loss.item(),psnr))
        g_lr_schedule.step()
        d_lr_schedule.step()

        train_mean_loss=train_mean_loss/len(train_bar)
        train_mean_psnr=train_mean_psnr/len(train_bar)
        writer.add_scalar('train_loss',train_mean_loss,epoch)
        writer.add_scalar('train_psnr',train_mean_psnr,epoch)
        print('\n train_loss:{} train_psnr:{}\n'.format(train_mean_loss,train_mean_psnr))
        if epoch%opts.save_epoch==0:
            g.eval()
            val_bar=tqdm(val_dataloader)
            val_mean_loss = 0
            val_mean_psnr=0
            val_mean_ssim=0
            with torch.no_grad():
                for lr,hr in val_bar:
                    lr, hr = lr.to(device), hr.to(device)
                    res=g(lr)
                    loss=g_loss_func(res,hr)

                    val_mean_loss+=loss.item()
                    psnr=peak_signal_noise_ratio(hr.cpu().numpy(),res.cpu().numpy())
                    ssim=structural_similarity(hr.cpu().numpy().squeeze().transpose(2,1,0),res.cpu().numpy().squeeze().transpose(2,1,0),multichannel=True)

                    val_mean_psnr +=psnr
                    val_mean_ssim +=ssim
                    val_bar.set_description(desc='Val Epoch:{} Loss:{} PSNR:{} SSIM:{}'.format(epoch,loss.item(),psnr,ssim))
                    # 每次只输出最后一组的验证结果
                    writer.add_images('SRResGan/epoch_' + str(epoch) + '_lr',
                                      lr, epoch)
                    writer.add_images('SRResGan/epoch_' + str(epoch) + '_res',
                                      res, epoch)
                    writer.add_images('SRResGan/epoch_' + str(epoch) + '_hr',
                                      hr, epoch)
                val_mean_loss/=len(val_bar)
                val_mean_psnr/=len(val_bar)
                val_mean_ssim/=len(val_bar)
                writer.add_scalar('val_mean_loss',val_mean_loss,epoch)
                writer.add_scalar('val_mean_psnr',val_mean_psnr,epoch)
                writer.add_scalar('val_mean_ssim',val_mean_ssim,epoch)
                print('\n val_mean_loss:{} val_psnr:{} val_ssim:{}\n'.format(val_mean_loss, val_mean_psnr,val_mean_ssim))
            d={
                'g_weights':g.state_dict(),
                'd_weights':d.state_dict(),
                'epoch':epoch,
                'g_optimizer':g_optimizer.state_dict(),
                'd_optimizer':d_optimizer.state_dict(),
                'g_schedule':g_lr_schedule.state_dict(),
                'd_schedule':d_lr_schedule.state_dict(),
               }
            torch.save(d,'./weights/SRResGan_{}.pth'.format(epoch))
    writer.close()

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--data_path_root','-dpr',default='../datasets/bsds500',type=str)
    args.add_argument('--batchsize','-bs',default=2,type=int)
    args.add_argument('--seed',default=1314,type=int)
    args.add_argument('--weights','-w',default=None,type=str)
    args.add_argument('--logs_dir','-ld',default='./logs',type=str)
    args.add_argument('--scale_factor','-sf',default=4,type=int)
    args.add_argument('--learning_rate','-lr',default=0.0001,type=float)
    args.add_argument('--epoch','-e',default=500,type=int)
    args.add_argument('--save_epoch','-se',default=1,type=int)
    args.add_argument('--workers','-wks',default=None,type=int)

    opts=args.parse_args()
    main(opts)