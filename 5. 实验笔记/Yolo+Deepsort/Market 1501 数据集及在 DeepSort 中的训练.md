> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [my.oschina.net](https://my.oschina.net/u/4883419/blog/4990739)

软硬件环境

*   ubuntu 18.04 64bit
    
*   GTX 1070Ti
    
*   anaconda with python 3.7
    
*   pytorch 1.6
    
*   cuda 10.1
    

Market 1501 数据集
---------------

`Market-1501`数据集是在清华大学校园中采集，在夏天拍摄，于 2015 年构建并公开。它包括由 6 个摄像头 (其中 5 个高清摄像头和 1 个低分辨率摄像头) 拍摄到的 1501 个行人、32668 个检测到的行人矩形框。每个行人至少有 2 个摄像头捕捉到，并且在一个摄像头中可能具有多张图像。训练集有 751 人，包含 12936 张图像，平均每个人有 17.2 张训练数据；测试集有 750 人，包含 19732 张图像，平均每个人有 26.3 张测试数据。

数据集目录结构

```
Market-1501-v15.09.15├── bounding_box_test├── bounding_box_train├── gt_bbox├── gt_query├── query└── readme.txt
```

包含四个文件夹

*   `bounding_box_test`: 测试集
    
*   `bounding_box_train`: 训练集
    
*   `query`: 共有 750 个身份。每个摄像机随机选择一个查询图像
    
*   `gt_query`: 包含实际标注
    
*   `gt_bbox`: 手绘边框，主要用于判断自动检测器 `DPM`边界框是否良好
    

图片命名规则

以`0001_c1s1_000151_01.jpg`为例

*   0001 表示每个人的标签编号，从 0001 到 1501，共有 1501 个人
    
*   `c1`表示第一个摄像头 ( `c`是 `camera`)，共有 6 个摄像头
    
*   `s1` 表示第一个录像片段 ( `s`是 `sequence`)，每个摄像机都有多个录像片段
    
*   000151 表示 `c1s1`的第 000151 帧图片，视频帧率 fps 为 25
    
*   01 表示 `c1s1_001051`这一帧上的第 1 个检测框，由于采用 `DPM`自动检测器，每一帧上的行人可能会有多个，相应的标注框也会有多个。00 则表示手工标注框
    

数据集下载地址：

链接：https://pan.baidu.com/s/1i9aiZx-EC3fjhn3uWTKZjw  
提取码：`up8x`

deepsort 模型训练
-------------

前文 《基于 YOLOv5 和 DeepSort 的目标跟踪》 https://xugaoxiang.com/2020/10/17/yolov5-deepsort-pytorch/ 介绍过利用`YOLOv5`和`DeepSort`来实现目标的检测及跟踪。现在我们使用`Market 1501`数据集来训练跟踪器模型。

至于`YOLOv5`检测模型的训练，参考前面的博文 YOLOv5 模型训练。我们使用原作者提供的`yolov5s.pt`就可行。

依赖环境就不再说了，参考前文

```
git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.gitcd Yolov5_DeepSort_Pytorch/deep_sort/deep_sort/deep
```

接下来将数据集`Market`拷贝到`Yolov5_DeepSort_Pytorch/deep_sort/deep_sort/deep`下然后解压，数据集存放的位置是随意的，可以通过参数`--data-dir`指定。

针对原项目中的训练代码`train.py`，需要做点修改

将

```
train_dir = os.path.join(root,"train")test_dir = os.path.join(root,"test")
```

改成

```
train_dir = os.path.join(root,"")test_dir = os.path.join(root,"")
```

然后将数据集中的文件夹`bounding_box_train`重命名为`train`，`bounding_box_test`重命名为`test`。不然的话，训练的时候就会报下面 2 个错

```
Traceback (most recent call last):  File "train.py", line 43, in <module>    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),  File "/home/xugaoxiang/anaconda3/envs/deepsort/lib/python3.7/site-packages/torchvision/datasets/folder.py", line 208, in __init__    is_valid_file=is_valid_file)  File "/home/xugaoxiang/anaconda3/envs/deepsort/lib/python3.7/site-packages/torchvision/datasets/folder.py", line 100, in __init__    raise RuntimeError(msg)RuntimeError: Found 0 files in subfolders of: data/trainSupported extensions are: .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif,.tiff,.webp
```

和

```
Traceback (most recent call last):  File "train.py", line 43, in <module>    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),  File "/home/xugaoxiang/anaconda3/envs/deepsort/lib/python3.7/site-packages/torchvision/datasets/folder.py", line 208, in __init__    is_valid_file=is_valid_file)  File "/home/xugaoxiang/anaconda3/envs/deepsort/lib/python3.7/site-packages/torchvision/datasets/folder.py", line 94, in __init__    classes, class_to_idx = self._find_classes(self.root)  File "/home/xugaoxiang/anaconda3/envs/deepsort/lib/python3.7/site-packages/torchvision/datasets/folder.py", line 123, in _find_classes    classes = [d.name for d in os.scandir(dir) if d.is_dir()]FileNotFoundError: [Errno 2] No such file or directory: 'Market-1501-v15.09.15/train'
```

最后，还需要改个地方，编辑`model.py`，将

```
def __init__(self, num_classes=751 ,reid=False):
```

改成

```
def __init__(self, num_classes=2 ,reid=False):
```

然后就可以开始训练了

```
python train.py --data-dir Market-1501-v15.09.15
```

![](https://oscimg.oschina.net/oscnet/ff8a598c-011b-43a0-8328-1d929716c6bf.png)

deepsort_market_pytorch

![](https://oscimg.oschina.net/oscnet/33a1f6f7-3fe9-464f-852c-e2acb09ff533.png)

deepsort_market_pytorch

训练结束后，会在`checkpoint`下生成模型文件`ckpt.t7`，找个视频，测试一下

deepsort_market_pytorch  

参考资料  

-------

*   https://xugaoxiang.com/2020/10/17/yolov5-deepsort-pytorch/
    
*   https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch
    
*   https://github.com/ZQPei/deep_sort_pytorch
    
*   https://github.com/ZQPei/deep_sort_pytorch/issues/134
    
*   https://github.com/ZQPei/deep_sort_pytorch/issues/105
    

本文分享自微信公众号 - 迷途小书童的 Note（Dev_Club）。  
如有侵权，请联系 support@oschina.cn 删除。  
本文参与 “[OSC 源创计划](https://www.oschina.net/sharing-plan)”，欢迎正在阅读的你也加入，一起分享。