# SRGan 论文复现
## SRResNet
见train.ipynb 一直运行到倒数第二行即可。~~使用ipynb的数据集可能train的不是很好~~
建议使用coco2014数据集的验证集4w张图片作为训练集或者VOC1012 10k做训练集也可以，bsd100作为验证集。
结果虽然是psnr=24~~差不多吧~~，但自己测试效果还行。
如果只使用bsd500训练，训练轮次太少效果不大行。

## SRGan
将SRRes_train.ipynb的倒数第二行改为运行最后一行
因为要训练判别器所以PSNR是先下降再上升的，但不会超过SRResNet的PSNR~~所以不用一直train期待会变更好~~。
SRGan的最终训练效果上限就是SRResNet的PSNR，Gan只是对SRResNet的一种微调，增加细节。

训练的结果如下：分别是插值，训练245 epoch 的gan和原始的SRResNet生成的图片

![image_lb](https://github.com/dummerchen/SRResGan/blob/master/results/demo/bear/lr_b.png)
![image_gan](https://github.com/dummerchen/SRResGan/blob/master/results/demo/bear/res_0.png)
![image_res](https://github.com/dummerchen/SRResGan/blob/master/results/demo/bear/res_1.png)


![image_lb](https://github.com/dummerchen/SRResGan/blob/master/results/demo/keqing/lr_b.png)
![image_gan](https://github.com/dummerchen/SRResGan/blob/master/results/demo/keqing/res_0.png)
![image_res](https://github.com/dummerchen/SRResGan/blob/master/results/demo/keqing/res_1.png)

可以发现gan的颜色虽然有点不对（SRResNet的PSNR低导致的）但是细节比SRResNet生成的更好
仔细看水的波纹和熊身上的毛还有刻晴背后的虚影
