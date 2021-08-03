> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_44791964/article/details/106214657)

### 睿智的目标检测 30——Pytorch 搭建 YoloV4 目标检测平台

*   [学习前言](#_2)
*   [什么是 YOLOV4](#YOLOV4_5)
*   [代码下载](#_25)
*   [YOLOV4 改进的部分（不完全）](#YOLOV4_28)
*   [YOLOV4 结构解析](#YOLOV4_48)
*   *   [1、主干特征提取网络 Backbone](#1Backbone_52)
    *   [2、特征金字塔](#2_264)
    *   [3、YoloHead 利用获得到的特征进行预测](#3YoloHead_373)
    *   [4、预测结果的解码](#4_466)
    *   [5、在原图上进行绘制](#5_560)
*   [YOLOV4 的训练](#YOLOV4_563)
*   *   [1、YOLOV4 的改进训练技巧](#1YOLOV4_564)
    *   *   [a)、Mosaic 数据增强](#aMosaic_565)
        *   [b)、Label Smoothing 平滑](#bLabel_Smoothing_771)
        *   [c)、CIOU](#cCIOU_790)
        *   [d)、学习率余弦退火衰减](#d_866)
    *   [2、loss 组成](#2loss_877)
    *   *   [a)、计算 loss 所需参数](#aloss_878)
        *   [b)、y_pre 是什么](#by_pre_886)
        *   [c)、y_true 是什么。](#cy_true_890)
        *   [d)、loss 的计算过程](#dloss_892)
*   [训练自己的 YOLOV4 模型](#YOLOV4_1175)

学习前言
====

也做了一下 pytorch 版本的。  
![](https://img-blog.csdnimg.cn/20190723165901974.jpg#pic_center)

什么是 YOLOV4
==========

![](https://img-blog.csdnimg.cn/20200509111027627.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
YOLOV4 是 YOLOV3 的改进版，**在 YOLOV3 的基础上结合了非常多的小 Tricks。**  
尽管没有目标检测上革命性的改变，但是 YOLOV4 依然很好的结合了速度与精度。  
根据上图也可以看出来，**YOLOV4 在 YOLOV3 的基础上，在 FPS 不下降的情况下，mAP 达到了 44，提高非常明显。**

YOLOV4 整体上的检测思路和 YOLOV3 相比相差并不大，都是使用三个特征层进行分类与回归预测。

**请注意！**

**强烈建议在学习 YOLOV4 之前学习 YOLOV3，因为 YOLOV4 确实可以看作是 YOLOV3 结合一系列改进的版本！**

**强烈建议在学习 YOLOV4 之前学习 YOLOV3，因为 YOLOV4 确实可以看作是 YOLOV3 结合一系列改进的版本！**

**强烈建议在学习 YOLOV4 之前学习 YOLOV3，因为 YOLOV4 确实可以看作是 YOLOV3 结合一系列改进的版本！**

（重要的事情说三遍！）

YOLOV3 可参考该博客：  
[https://blog.csdn.net/weixin_44791964/article/details/105310627](https://blog.csdn.net/weixin_44791964/article/details/105310627)

代码下载
====

[https://github.com/bubbliiiing/yolov4-pytorch](https://github.com/bubbliiiing/yolov4-pytorch)  
喜欢的可以给个 star 噢！

YOLOV4 改进的部分（不完全）
=================

**1、主干特征提取网络：DarkNet53 => CSPDarkNet53**

**2、特征金字塔：SPP，PAN**

**3、分类回归层：YOLOv3（未改变）**

**4、训练用到的小技巧：Mosaic 数据增强、Label Smoothing 平滑、CIOU、学习率余弦退火衰减**

**5、激活函数：使用 Mish 激活函数**

**以上并非全部的改进部分，还存在一些其它的改进，由于 YOLOV4 使用的改进实在太多了，很难完全实现与列出来，这里只列出来了一些我比较感兴趣，而且非常有效的改进。**

还有一个重要的事情：  
**论文中提到的 SAM，作者自己的源码也没有使用。**

**还有其它很多的 tricks，不是所有的 tricks 都有提升，我也没法实现全部的 tricks**。

整篇 BLOG 会结合 YOLOV3 与 YOLOV4 的差别进行解析

YOLOV4 结构解析
===========

**为方便理解，本文将所有通道数都放到了最后一维度。  
为方便理解，本文将所有通道数都放到了最后一维度。  
为方便理解，本文将所有通道数都放到了最后一维度。**

1、主干特征提取网络 Backbone
-------------------

当输入是 416x416 时，特征结构如下：  
![](https://img-blog.csdnimg.cn/20200512144007178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
当输入是 608x608 时，特征结构如下：  
![](https://img-blog.csdnimg.cn/20200509213429592.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
主干特征提取网络 Backbone 的改进点有两个：  
**a). 主干特征提取网络：DarkNet53 => CSPDarkNet53  
b). 激活函数：使用 Mish 激活函数**

如果大家对 YOLOV3 比较熟悉的话，应该知道 Darknet53 的结构，其由一系列残差网络结构构成。在 Darknet53 中，其存在 **resblock_body 模块**，其由一次**下采样**和**多次残差结构的堆叠**构成，Darknet53 便是由 **resblock_body 模块组合而成**。

而在 YOLOV4 中，其对该部分进行了一定的修改。  
1、其一是**将 DarknetConv2D 的激活函数由 LeakyReLU 修改成了 Mish**，卷积块由 **DarknetConv2D_BN_Leaky 变成了 DarknetConv2D_BN_Mish**。  
Mish 函数的公式与图像如下：  
M i s h = x × t a n h ( l n ( 1 + e x ) ) Mish=x \times tanh(ln(1+e^x)) Mish=x×tanh(ln(1+ex))  
![](https://img-blog.csdnimg.cn/20200509115101964.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
2、其二是将 resblock_body 的结构进行修改，**使用了 CSPnet 结构。此时 YOLOV4 当中的 Darknet53 被修改成了 CSPDarknet53**。  
![](https://img-blog.csdnimg.cn/20200509113651540.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
CSPnet 结构并不算复杂，就是将原来的残差块的堆叠进行了一个拆分，拆成左右两部分：  
**主干部分继续进行原来的残差块的堆叠**；  
**另一部分则像一个残差边一样，经过少量处理直接连接到最后。**  
因此可以认为 **CSP 中存在一个大的残差边。**

```
#---------------------------------------------------#
#   CSPdarknet的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)  
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)

            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x

```

全部实现代码为：

```
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict

#-------------------------------------------------#
#   MISH激活函数
#-------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#-------------------------------------------------#
#   卷积块
#   CONV+BATCHNORM+MISH
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块
#---------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None, residual_activation=nn.Identity()):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x+self.block(x)

#---------------------------------------------------#
#   CSPdarknet的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)  
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)

            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x

class CSPDarkNet(nn.Module):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    model = CSPDarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model

```

2、特征金字塔
-------

当输入是 416x416 时，特征结构如下：  
![](https://img-blog.csdnimg.cn/20200512144007178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
当输入是 608x608 时，特征结构如下：  
![](https://img-blog.csdnimg.cn/20200509213429592.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
在特征金字塔部分，YOLOV4 结合了两种改进:  
**a). 使用了 SPP 结构。  
b). 使用了 PANet 结构。**  
如上图所示，除去 CSPDarknet53 和 Yolo Head 的结构外，都是特征金字塔的结构。  
1、SPP 结构参杂在**对 CSPdarknet53 的最后一个特征层的卷积里**，在**对 CSPdarknet53 的最后一个特征层进行三次 DarknetConv2D_BN_Leaky 卷积后**，**分别利用四个不同尺度的最大池化进行处理**，最大池化的**池化核大小分别为 13x13、9x9、5x5、1x1（1x1 即无处理）**

```
#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

```

其可以**它能够极大地增加感受野，分离出最显著的上下文特征**。  
![](https://img-blog.csdnimg.cn/20200509215346310.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
2、PANet 是 2018 的一种实例分割算法，其具体结构由反复提升特征的意思。  
![](https://img-blog.csdnimg.cn/20200509214653559.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
上图为原始的 PANet 的结构，可以看出来其具有**一个非常重要的特点就是特征的反复提取**。  
在（a）里面是传统的特征金字塔结构，在完成特征金字塔从**下到上的特征提取后**，还需要**实现（b）中从上到下的特征提取。**

而在 YOLOV4 当中，其主要是在**三个有效特征层上使用了 PANet 结构。**  
![](https://img-blog.csdnimg.cn/20200509215322866.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
实现代码如下：

```
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config
        #  backbone
        self.backbone = darknet53(None)

        self.conv1 = make_three_conv([512,1024],1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.conv_for_P4 = conv2d(512,256,1)
        self.make_five_conv1 = make_five_conv([256, 512],512)

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = conv2d(256,128,1)
        self.make_five_conv2 = make_five_conv([128, 256],256)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.yolo_head3 = yolo_head([256, final_out_filter2],128)

        self.down_sample1 = conv2d(128,256,3,stride=2)
        self.make_five_conv3 = make_five_conv([256, 512],512)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.yolo_head2 = yolo_head([512, final_out_filter1],256)


        self.down_sample2 = conv2d(256,512,3,stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024],1024)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512)


    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4,P5_upsample],axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3,P4_upsample],axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample,P4],axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample,P5],axis=1)
        P5 = self.make_five_conv4(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2

```

3、YoloHead 利用获得到的特征进行预测
-----------------------

当输入是 416x416 时，特征结构如下：  
![](https://img-blog.csdnimg.cn/20200512144007178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
当输入是 608x608 时，特征结构如下：  
![](https://img-blog.csdnimg.cn/20200509213429592.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
1、在特征利用部分，YoloV4 提取**多特征层进行目标检测**，一共**提取三个特征层**，分别位于中间层，中下层，底层，三个特征层的 shape 分别为 (76,76,256)、(38,38,512)、(19,19,1024)。

2、输出层的 shape 分别为 (**19,19,75**)，(**38,38,75**)，(**76,76,75**)，**最后一个维度为 75 是因为该图是基于 voc 数据集的，它的类为 20 种，YoloV4 只有针对每一个特征层存在 3 个先验框，所以最后维度为 3x25；  
如果使用的是 coco 训练集，类则为 80 种，最后的维度应该为 255 = 3x85**，三个特征层的 shape 为 (**19,19,255**)，(**38,38,255**)，(**76,76,255**)

实现代码如下：

```
#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config
        #  backbone
        self.backbone = darknet53(None)

        self.conv1 = make_three_conv([512,1024],1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.conv_for_P4 = conv2d(512,256,1)
        self.make_five_conv1 = make_five_conv([256, 512],512)

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = conv2d(256,128,1)
        self.make_five_conv2 = make_five_conv([128, 256],256)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.yolo_head3 = yolo_head([256, final_out_filter2],128)

        self.down_sample1 = conv2d(128,256,3,stride=2)
        self.make_five_conv3 = make_five_conv([256, 512],512)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.yolo_head2 = yolo_head([512, final_out_filter1],256)


        self.down_sample2 = conv2d(256,512,3,stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024],1024)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512)


    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4,P5_upsample],axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3,P4_upsample],axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample,P4],axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample,P5],axis=1)
        P5 = self.make_five_conv4(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2

```

4、预测结果的解码
---------

由第二步我们可以获得三个特征层的预测结果，shape 分别为 (**N,19,19,255**)，(**N,38,38,255**)，(**N,76,76,255**) 的数据，对应每个图分为 19x19、38x38、76x76 的网格上 3 个预测框的位置。

但是这个预测结果并不对应着最终的预测框在图片上的位置，还需要解码才可以完成。

此处要讲一下 yolo3 的预测原理，yolo3 的 3 个特征层分别将整幅图分为 19x19、38x38、76x76 的网格，每个网络点负责一个区域的检测。

我们知道特征层的预测结果对应着三个预测框的位置，我们先将其 reshape 一下，其结果为 (**N,19,19,3,85**)，(**N,38,38,3,85**)，(**N,76,76,3,85**)。

最后一个维度中的 85 包含了 4+1+80，**分别代表 x_offset、y_offset、h 和 w、置信度、分类结果。**

yolo3 的解码过程就是**将每个网格点加上它对应的 x_offset 和 y_offset**，加完后的结果就是**预测框的中心**，然后**再利用 先验框和 h、w 结合 计算出预测框的长和宽**。这样就能得到整个预测框的位置了。

![](https://img-blog.csdnimg.cn/20191120215015351.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
当然得到最终的预测结构后还要进行得分排序与非极大抑制筛选  
这一部分基本上是所有目标检测通用的部分。不过该项目的处理方式与其它项目不同。其对于每一个类进行判别。  
**1、取出每一类得分大于 self.obj_threshold 的框和得分。  
2、利用框的位置和得分进行非极大抑制。**

实现代码如下，当调用 yolo_eval 时，就会对每个特征层进行解码：

```
class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        # input为bs,3*(1+4+num_classes),13,13

        # 一共多少张图片
        batch_size = input.size(0)
        # 13，13
        input_height = input.size(2)
        input_width = input.size(3)

        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        # 416/13 = 32
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        # 把先验框的尺寸调整成特征层大小的形式
        # 计算出先验框在特征层上对应的宽高
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]

        # bs,3*(5+num_classes),13,13 -> bs,3,13,13,(5+num_classes)
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角 batch_size,3,13,13
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_width, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
        
        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # 用于将输出调整为相对于416x416的大小
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data

```

5、在原图上进行绘制
----------

通过第四步，我们可以获得预测框在原图上的位置，而且这些预测框都是经过筛选的。这些筛选后的框可以直接绘制在图片上，就可以获得结果了。

YOLOV4 的训练
==========

1、YOLOV4 的改进训练技巧
----------------

### a)、Mosaic 数据增强

Yolov4 的 mosaic 数据增强参考了 CutMix 数据增强方式，理论上具有一定的相似性！  
**CutMix 数据增强方式利用两张图片进行拼接。**  
![](https://img-blog.csdnimg.cn/20200508145223449.png#pic_center)  
但是 mosaic 利用了四张图片，**根据论文所说其拥有一个巨大的优点是丰富检测物体的背景！且在 BN 计算的时候一下子会计算四张图片的数据！**  
就像下图这样：  
![](https://img-blog.csdnimg.cn/20200508145439305.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
实现思路如下：  
1、每次读取四张图片。

![](https://img-blog.csdnimg.cn/20200508155913308.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
![](https://img-blog.csdnimg.cn/20200508155925719.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
![](https://img-blog.csdnimg.cn/20200508155933988.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
![](https://img-blog.csdnimg.cn/20200508155946892.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
2、分别对四张图片进行翻转、缩放、色域变化等，并且按照四个方向位置摆好。  
![](https://img-blog.csdnimg.cn/20200508160000580.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
![](https://img-blog.csdnimg.cn/20200508160015434.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
![](https://img-blog.csdnimg.cn/20200508160028438.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
![](https://img-blog.csdnimg.cn/20200508160038303.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
3、进行图片的组合和框的组合  
![](https://img-blog.csdnimg.cn/20200508160045845.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)

```
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def merge_bboxes(bboxes, cutx, cuty):

    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue
                
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue
                
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox

def get_random_data(annotation_line, input_shape, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    h, w = input_shape
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1-min(min_offset_x,min_offset_y)
    scale_high = scale_low+0.2

    image_datas = [] 
    box_datas = []
    index = 0

    place_x = [0,0,int(w*min_offset_x),int(w*min_offset_x)]
    place_y = [0,int(h*min_offset_y),int(w*min_offset_y),0]
    for line in annotation_line:
        # 每一行进行分割
        line_content = line.split()
        # 打开图片
        image = Image.open(line_content[0])
        image = image.convert("RGB") 
        # 图片的大小
        iw, ih = image.size
        # 保存框的位置
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
        
        # image.save(str(index)+".jpg")
        # 是否翻转图片
        flip = rand()<.5
        if flip and len(box)>0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0,2]] = iw - box[:, [2,0]]

        # 对输入进来的图片进行缩放
        new_ar = w/h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # 进行色域变换
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image = hsv_to_rgb(x)

        image = Image.fromarray((image*255).astype(np.uint8))
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)/255

        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
        
        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        
        image_datas.append(image_data)
        box_datas.append(box_data)

        img = Image.fromarray((image_data*255).astype(np.uint8))
        for j in range(len(box_data)):
            thickness = 3
            left, top, right, bottom  = box_data[j][0:4]
            draw = ImageDraw.Draw(img)
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(255,255,255))
        img.show()

    
    # 将图片分割，放在一起
    cutx = np.random.randint(int(w*min_offset_x), int(w*(1 - min_offset_x)))
    cuty = np.random.randint(int(h*min_offset_y), int(h*(1 - min_offset_y)))

    new_image = np.zeros([h,w,3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes

```

### b)、Label Smoothing 平滑

标签平滑的思想很简单，具体公式如下：

```
new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes

```

当 label_smoothing 的值为 0.01 得时候，公式变成如下所示：

```
new_onehot_labels = y * (1 - 0.01) + 0.01 / num_classes

```

其实 Label Smoothing 平滑就是将标签进行一个平滑，原始的标签是 0、1，在平滑后变成 0.005(如果是二分类)、0.995，**也就是说对分类准确做了一点惩罚，让模型不可以分类的太准确，太准确容易过拟合。**

实现代码如下：

```
#---------------------------------------------------#
#   平滑标签
#---------------------------------------------------#
def smooth_labels(y_true, label_smoothing,num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

```

### c)、CIOU

IoU 是比值的概念，对目标物体的 scale 是不敏感的。**然而常用的 BBox 的回归损失优化和 IoU 优化不是完全等价的，寻常的 IoU 无法直接优化没有重叠的部分。**

于是有人提出直接使用 IOU 作为回归优化 loss，CIOU 是其中非常优秀的一种想法。

CIOU 将目标与 anchor 之间的距离，重叠率、尺度以及惩罚项都考虑进去，**使得目标框回归变得更加稳定，不会像 IoU 和 GIoU 一样出现训练过程中发散等问题。而惩罚因子把预测框长宽比拟合目标框的长宽比考虑进去。**

![](https://img-blog.csdnimg.cn/20200425144646161.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
CIOU 公式如下  
C I O U = I O U − ρ 2 ( b , b g t ) c 2 − α v CIOU = IOU - \frac{\rho^2(b,b^{gt})}{c^2} - \alpha v CIOU=IOU−c2ρ2(b,bgt)​−αv  
其中， ρ 2 ( b , b g t ) \rho^2(b,b^{gt}) ρ2(b,bgt) 分别**代表了预测框和真实框的中心点的欧式距离。 c 代表的是能够同时包含预测框和真实框的最小闭包区域的对角线距离。**

而 α \alpha α和 v v v 的公式如下  
α = v 1 − I O U + v \alpha = \frac{v}{1-IOU+v} α=1−IOU+vv​  
v = 4 π 2 ( a r c t a n w g t h g t − a r c t a n w h ) 2 v = \frac{4}{\pi ^2}(arctan\frac{w^{gt}}{h^{gt}}-arctan\frac{w}{h})^2 v=π24​(arctanhgtwgt​−arctanhw​)2  
把 1-CIOU 就可以得到相应的 LOSS 了。  
L O S S C I O U = 1 − I O U + ρ 2 ( b , b g t ) c 2 + α v LOSS_{CIOU} = 1 - IOU + \frac{\rho^2(b,b^{gt})}{c^2} + \alpha v LOSSCIOU​=1−IOU+c2ρ2(b,bgt)​+αv

```
def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / (union_area + 1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / (enclose_diagonal + 1e-7)
    
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/b1_wh[..., 1]) - torch.atan(b2_wh[..., 0]/b2_wh[..., 1])), 2)
    alpha = v / (1.0 - iou + v)
    ciou = ciou - alpha * v
    return ciou

```

### d)、学习率余弦退火衰减

余弦退火衰减法，学习率会先上升再下降，这是退火优化法的思想。（关于什么是退火算法可以百度。）

上升的时候使用线性上升，下降的时候模拟 cos 函数下降。执行多次。

效果如图所示：  
![](https://img-blog.csdnimg.cn/20200519151307934.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
pytorch 有直接实现的函数，可直接调用。

```
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

```

2、loss 组成
---------

### a)、计算 loss 所需参数

在计算 loss 的时候，实际上是 y_pre 和 y_true 之间的对比：  
y_pre 就是一幅**图像经过网络之后的输出，内部含有三个特征层的内容；其需要解码才能够在图上作画**  
y_true 就是一个真实图像中，它的**每个真实框对应的 (19,19)、(38,38)、(76,76) 网格上的偏移位置、长宽与种类。其仍需要编码才能与 y_pred 的结构一致**  
实际上 y_pre 和 y_true 内容的 shape 都是  
(batch_size,19,19,3,85)  
(batch_size,38,38,3,85)  
(batch_size,76,76,3,85)

### b)、y_pre 是什么

网络**最后输出的内容就是三个特征层每个网格点对应的预测框及其种类**，即三个特征层分别对应着图片被分为不同 size 的网格后，**每个网格点上三个先验框对应的位置、置信度及其种类。**  
对于**输出的 y1、y2、y3 而言，[…, : 2] 指的是相对于每个网格点的偏移量，[…, 2: 4] 指的是宽和高，[…, 4: 5] 指的是该框的置信度，[…, 5: ] 指的是每个种类的预测概率。**  
现在的 y_pre 还是没有解码的，解码了之后才是真实图像上的情况。

### c)、y_true 是什么。

y_true 就是一个真实图像中，它的**每个真实框对应的 (19,19)、(38,38)、(76,76) 网格上的偏移位置、长宽与种类。其仍需要编码才能与 y_pred 的结构一致**

### d)、loss 的计算过程

在得到了 y_pre 和 y_true 后怎么对比呢？不是简单的减一下!

loss 值需要对三个特征层进行处理，这里以最小的特征层为例。  
**1、利用 y_true 取出该特征层中真实存在目标的点的位置 (m,19,19,3,1) 及其对应的种类(m,19,19,3,80)。  
2、将 prediction 的预测值输出进行处理，得到 reshape 后的预测值 y_pre，shape 为 (m,19,19,3,85)。还有解码后的 xy，wh。  
3、对于每一幅图，计算其中所有真实框与预测框的 IOU，如果某些预测框和真实框的重合程度大于 0.5，则忽略。  
4、计算 ciou 作为回归的 loss，这里只计算正样本的回归 loss。  
5、计算置信度的 loss，其有两部分构成，第一部分是实际上存在目标的，预测结果中置信度的值与 1 对比；第二部分是实际上不存在目标的，在第四步中得到其最大 IOU 的值与 0 对比。  
6、计算预测种类的 loss，其计算的是实际上存在目标的，预测类与真实类的差距。**

其实际上计算的总的 loss 是三个 loss 的和，这三个 loss 分别是：

*   实际存在的框，**CIOU LOSS**。
*   实际存在的框，**预测结果中置信度的值与 1 对比；实际不存在的框，预测结果中置信度的值与 0 对比，该部分要去除被忽略的不包含目标的框**。
*   实际存在的框，**种类预测结果与实际结果的对比**。

其实际代码如下：

```
#---------------------------------------------------#
#   平滑标签
#---------------------------------------------------#
def smooth_labels(y_true, label_smoothing,num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / (union_area + 1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / (enclose_diagonal + 1e-7)
    
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/b1_wh[..., 1]) - torch.atan(b2_wh[..., 0]/b2_wh[..., 1])), 2)
    alpha = v / (1.0 - iou + v)
    ciou = ciou - alpha * v
    return ciou

def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def MSELoss(pred,target):
    return (pred-target)**2

def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, label_smooth=0, cuda=True):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        self.label_smooth = label_smooth

        self.ignore_threshold = 0.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0
        self.cuda = cuda

    def forward(self, input, targets=None):
        # input为bs,3*(5+num_classes),13,13
        
        # 一共多少张图片
        bs = input.size(0)
        # 特征层的高
        in_h = input.size(2)
        # 特征层的宽
        in_w = input.size(3)

        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        # 把先验框的尺寸调整成特征层大小的形式
        # 计算出先验框在特征层上对应的宽高
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        # bs,3*(5+num_classes),13,13 -> bs,3,13,13,(5+num_classes)
        prediction = input.view(bs, int(self.num_anchors/3),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        # 对prediction预测进行调整
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # 找到哪些先验框内部包含物体
        mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.get_target(targets, scaled_anchors,in_w, in_h,self.ignore_threshold)

        noobj_mask, pred_boxes_for_ciou = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)

        if self.cuda:
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            box_loss_scale_x, box_loss_scale_y= box_loss_scale_x.cuda(), box_loss_scale_y.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            pred_boxes_for_ciou = pred_boxes_for_ciou.cuda()
            t_box = t_box.cuda()

        box_loss_scale = 2-box_loss_scale_x*box_loss_scale_y
        #  losses.
        ciou = (1 - box_ciou( pred_boxes_for_ciou[mask.bool()], t_box[mask.bool()]))* box_loss_scale[mask.bool()]

        loss_loc = torch.sum(ciou / bs)
        loss_conf = torch.sum(BCELoss(conf, mask) * mask / bs) + \
                    torch.sum(BCELoss(conf, mask) * noobj_mask / bs)
                    
        # print(smooth_labels(tcls[mask == 1],self.label_smooth,self.num_classes))
        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], smooth_labels(tcls[mask == 1],self.label_smooth,self.num_classes))/bs)
        # print(loss_loc,loss_conf,loss_cls)
        loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc
        return loss, loss_conf.item(), loss_cls.item(), loss_loc.item()

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        # 计算一共有多少张图片
        bs = len(target)
        # 获得先验框
        anchor_index = [[0,1,2],[3,4,5],[6,7,8]][[13,26,52].index(in_w)]
        subtract_index = [0,3,6][[13,26,52].index(in_w)]
        # 创建全是0或者全是1的阵列
        mask = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        t_box = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, 4, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, self.num_classes, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        for b in range(bs):
            for t in range(target[b].shape[0]):
                # 计算出在特征层上的点位
                gx = target[b][t, 0] * in_w
                gy = target[b][t, 1] * in_h
                
                gw = target[b][t, 2] * in_w
                gh = target[b][t, 3] * in_h

                # 计算出属于哪个网格
                gi = int(gx)
                gj = int(gy)

                # 计算真实框的位置
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                
                # 计算出所有先验框的位置
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # 计算重合程度
                anch_ious = bbox_iou(gt_box, anchor_shapes)
               
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)
                if best_n not in anchor_index:
                    continue
                # Masks
                if (gj < in_h) and (gi < in_w):
                    best_n = best_n - subtract_index
                    # 判定哪些先验框内部真实的存在物体
                    noobj_mask[b, best_n, gj, gi] = 0
                    mask[b, best_n, gj, gi] = 1
                    # 计算先验框中心调整参数
                    tx[b, best_n, gj, gi] = gx
                    ty[b, best_n, gj, gi] = gy
                    # 计算先验框宽高调整参数
                    tw[b, best_n, gj, gi] = gw
                    th[b, best_n, gj, gi] = gh
                    # 用于获得xywh的比例
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][t, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][t, 3]
                    # 物体置信度
                    tconf[b, best_n, gj, gi] = 1
                    # 种类
                    tcls[b, best_n, gj, gi, int(target[b][t, 4])] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue
        t_box[...,0] = tx
        t_box[...,1] = ty
        t_box[...,2] = tw
        t_box[...,3] = th
        return mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self,prediction,target,scaled_anchors,in_w, in_h,noobj_mask):
        bs = len(target)
        anchor_index = [[0,1,2],[3,4,5],[6,7,8]][[13,26,52].index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]
        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
            int(bs*self.num_anchors/3), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
            int(bs*self.num_anchors/3), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        
        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)

            for t in range(target[i].shape[0]):
                gx = target[i][t, 0] * in_w
                gy = target[i][t, 1] * in_h
                gw = target[i][t, 2] * in_w
                gh = target[i][t, 3] * in_h
                gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0).type(FloatTensor)

                anch_ious = bbox_iou(gt_box, pred_boxes_for_ignore, x1y1x2y2=False)
                anch_ious = anch_ious.view(pred_boxes[i].size()[:3])
                noobj_mask[i][anch_ious>self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

```

训练自己的 YOLOV4 模型
===============

yolo4 整体的文件夹构架如下：  
![](https://img-blog.csdnimg.cn/20200511215524248.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
本文使用 VOC 格式进行训练。  
训练前将标签文件放在 VOCdevkit 文件夹下的 VOC2007 文件夹下的 Annotation 中。  
![](https://img-blog.csdnimg.cn/20191127171146309.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
训练前将图片文件放在 VOCdevkit 文件夹下的 VOC2007 文件夹下的 JPEGImages 中。  
![](https://img-blog.csdnimg.cn/20191127171315165.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
在训练前利用 voc2yolo3.py 文件生成对应的 txt。  
![](https://img-blog.csdnimg.cn/20200511215606296.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
再运行根目录下的 voc_annotation.py，运行前需要将 classes 改成你自己的 classes。

```
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

```

![](https://img-blog.csdnimg.cn/20200511215635790.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
就会生成对应的 2007_train.txt，每一行对应其图片位置及其真实框的位置。  
![](https://img-blog.csdnimg.cn/2020051121565488.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
在训练前需要修改 model_data 里面的 voc_classes.txt 文件，需要将 classes 改成你自己的 classes。  
![](https://img-blog.csdnimg.cn/20200511215711964.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)  
运行 train.py 即可开始训练。  
![](https://img-blog.csdnimg.cn/20200519160129375.png#pic_center)