> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [xugaoxiang.com](https://xugaoxiang.com/2020/10/17/yolov5-deepsort-pytorch/)

### 软硬件环境

*   windows 10 64bit
*   [pytorch](https://xugaoxiang.com/tag/pytorch/)
*   [yolov5](https://xugaoxiang.com/tag/yolov5/)
*   [deepsort](https://xugaoxiang.com/tag/deepsort/)

### 视频看这里

此处是`youtube`的播放链接，需要科学上网。喜欢我的视频，请记得订阅我的频道，打开旁边的小铃铛，点赞并分享，感谢您的支持。

### YOLOv5

前文 [YOLOv5 目标检测](https://xugaoxiang.com/2020/06/17/yolov5/) 和 [YOLOv5 模型训练](https://xugaoxiang.com/2020/07/02/yolov5-training/) 已经介绍过了`YOLOv5`相关的内容，在目标检测中效果不错。

### DeepSort

`SORT`算法的思路是将目标检测算法 (如`YOLO`) 得到的检测框与预测的跟踪框的`iou`(交并比) 输入到匈牙利算法中进行线性分配来关联帧间 `ID`。而`DeepSORT`算法则是将目标的外观信息加入到帧间匹配的计算中，这样在目标被遮挡但后续再次出现的情况下，还能正确匹配这个`ID`，从而减少`ID`的切换，达到持续跟踪的目的。

### 目标跟踪

项目地址 [https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)，使用的是`Pytorch`深度学习框架，联合`YOLOv5`和`DeepSort`两个目前很火且效果非常不错的算法工程，实现特定物体的目标跟踪。

```
git clone https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git
cd Yolov5_DeepSort_Pytorch
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

然后去下载权重文件，`YOLOv5`的权重文件放置在 `[yolov5](https://xugaoxiang.com/tag/yolov5/)/weights`文件夹下，`DeepSort`的权重文件`ckpt.t7`放置在`deep_sort/deep/checkpoint`文件夹下

下载链接，[百度网盘下载地址](https://pan.baidu.com/s/15c0i-EQuKiTXrOGp3ShxvQ)， 提取码：`u5v3`

找个测试视频，来看看效果吧

```
python track.py --source test.mp4
```

测试效果图

![](https://image.xugaoxiang.com/imgs/2020/10/8e5702f79cf8c677.png)

### 参考资料

*   [https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
*   [https://github.com/ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)
*   [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)