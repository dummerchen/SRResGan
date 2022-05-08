# SRGan 论文复现
## SRResNet
见SRResNet_train.ipynb 运行到倒数第二行即可
使用coco2014数据集的验证集作为训练集，bsd500的验证集作为测试集
结果psnr=28.3~~差不多吧~~，自己测试效果还行。
如果只使用bsd500训练，效果不行，始终psnr是22左右。

## SRGan
将SRRes_train.ipynb的倒数第二行改为运行最后一行
结果如下：分别是插值，gan，和训练的40个epoch的res生成的图片
![image_lb](https://github.com/dummerchen/SRResGan/blob/master/results/lr_b.png)
![image_gan](https://github.com/dummerchen/SRResGan/blob/master/results/res_0.png)
![image_res](https://github.com/dummerchen/SRResGan/blob/master/results/res_1.png)

可以发现gan的颜色虽然有点不对但是细节比SRResNet生成的更好~~仔细看水的波纹和桥的钢缆~~
