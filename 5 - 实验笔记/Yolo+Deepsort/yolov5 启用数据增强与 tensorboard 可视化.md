> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_41868104/article/details/114685071)

yolov5 启用数据增强与 tensorboard 可视化
------------------------------

一，yolov5 启用数据增强
---------------

1.data 目录下，有两个 hyp 的文件：data/hyp.scratch.yaml 和 data/hyp.finetune.yaml 具体内容如下：

```
# Hyperparameters for VOC fine-tuning
# python train.py --batch 64 --cfg '' --weights yolov5m.pt --data voc.yaml --img 512 --epochs 50
# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials


lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
momentum: 0.94  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
giou: 0.05  # GIoU loss gain
cls: 0.4  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 0.5  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.6  # image HSV-Value augmentation (fraction)
degrees: 1.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.6  # image scale (+/- gain)
shear: 1.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.01  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mixup: 0.2  # image mixup (probability)
```

2. 启用方法  
在 train.py 中添加指定，当然程序本身也会默认启动 hyp.scratch.yaml 这个，可以直接修改其内部参数，如果需要启用另一个，可以如图：  
![](https://img-blog.csdnimg.cn/20210312090050344.png#pic_center)  
训练时会在终端打印显示出相关参数设置情况

二，tensorboard 可视化
-----------------

良心 yolov5！感觉好多东西都直接写好了，调用即可。  
models/yolo.py 中，代码最底部作者将 tensorboard 代码注释了，启用即可。  
![](https://img-blog.csdnimg.cn/20210312090419365.png#pic_center)  
取消注释后，点击启动 tensorboard 会话。  
vs code 上出现如下提示：  
![](https://img-blog.csdnimg.cn/2021031209070383.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTg2ODEwNA==,size_16,color_FFFFFF,t_70#pic_center)  
直接点击使用当前目录时，无法查看效果。需要定位到 runs 文件夹。  
点击‘选择另一个文件夹’，找到 runs 文件夹。效果如图：  
![](https://img-blog.csdnimg.cn/20210312091014302.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTg2ODEwNA==,size_16,color_FFFFFF,t_70#pic_center)