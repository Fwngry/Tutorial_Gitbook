> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/92141879)

原文发表在：[语雀文档](https://link.zhihu.com/?target=https%3A//www.yuque.com/docs/share/0b38c575-705c-4c63-a707-0aefc74cedb9%23)
-----------------------------------------------------------------------------------------------------------------------

0. 前言
-----

本文为 **Darknet 框架下，利用官方 VOC 数据集的 yolov3 模型训练**，训练环境为：Ubuntu18.04 下的 GPU 训练，cuda 版本 10.0；cudnn 版本 7.6.5。经过一晚上的训练，模型 20 个类别的 mAP 达到 74%+。

主要模块：

*   **概述**
*   **源码编译**
*   **功能测试**
*   **模型训练**
*   **模型验证**

【概述】主要介绍 yolo 系列模型和 darknet 框架的关系、资源网站和数据集下载

【源码编译】主要是用官方源码和 make 命令编译出 Linux 下可执行文件，包括 cuda+cudnn 的设置

【功能测试】主要是用官网给出的 demo 和与训练模型来进行图片测试

【模型训练】主要是用 darknet+yolov3 模型训练 VOC 图像数据集（VOC2007+VOC2012） 【模型验证】即用训练好的模型，检测模型训练效果.

1. 概述
-----

官网：[https://pjreddie.com/](https://link.zhihu.com/?target=https%3A//pjreddie.com/)

![](https://pic1.zhimg.com/v2-74506e9a3056e4267ee3d16c79246d38_r.jpg)

1.1 Yolo
--------

Yolo 系列模型（v1~v3）在近年些来的目标检测领域，非常火热！Yolo 为：You only look once 的缩写，同时也表明了其特点，只需要一次即可完成从图像分割到检测，故其被称为 one-stage 系列模型的鼻祖。

> two-stage 类目标检测模型如 Fast R-CNN、Faster R-CNN

![](https://pic3.zhimg.com/v2-81a42591f84a083fc31961a78ac12d36_r.jpg)

1.2 Darknet
-----------

**yolo 系列就是深度学习领域中用于目标检测的模型（yolov1~v3），那么 darknet 是什么？两者关系如何？** darknet 是作者用 c 和 cuda 编写的用于深度学习模型训练的框架，支持 CPU 和 GPU 训练，是个非常简单轻量级框架，就和 Tensorflow,Mxnet，Pytorch,Caffe 一样，虽然功能没有它们多，不过小也有小的优势，如果你会 c 语言，可以对其进行充分地利用和改造，或者读读源码看一下其实现，也会收货满满！ **So, 总结一下：Darknet 是深度学习框架，yolov1~v3 是 yolo 系列的目标检测模型**

1.3 资源下载
--------

### 官网

[https://pjreddie.com/](https://link.zhihu.com/?target=https%3A//pjreddie.com/)

### 源码

Darknet 源码： [https://github.com/pjreddie/darknet](https://link.zhihu.com/?target=https%3A//github.com/pjreddie/darknet)

> Darknet 是用纯 c 和 cuda 编写的，要想用 darknet 来训练模型，最好用 Linux/Unix 系统，官方也提供了 python 的接口，如果是 Windows 系统，可以利用网友开源实现：[https://github.com/AlexeyAB/darknet](https://link.zhihu.com/?target=https%3A//github.com/AlexeyAB/darknet)

可以直接下载 zip 包 [darknet-master.zip](https://link.zhihu.com/?target=https%3A//www.yuque.com/attachments/yuque/0/2019/zip/216914/1573805361157-7fe55f8e-8acd-4db5-a065-78142b19d7bc.zip%3F_lake_card%3D%257B%2522uid%2522%253A%25221573805355200-0%2522%252C%2522src%2522%253A%2522https%253A%252F%252Fwww.yuque.com%252Fattachments%252Fyuque%252F0%252F2019%252Fzip%252F216914%252F1573805361157-7fe55f8e-8acd-4db5-a065-78142b19d7bc.zip%2522%252C%2522name%2522%253A%2522darknet-master.zip%2522%252C%2522size%2522%253A3672348%252C%2522type%2522%253A%2522application%252Fzip%2522%252C%2522ext%2522%253A%2522zip%2522%252C%2522progress%2522%253A%257B%2522percent%2522%253A0%257D%252C%2522status%2522%253A%2522done%2522%252C%2522percent%2522%253A0%252C%2522id%2522%253A%25224b5dX%2522%252C%2522refSrc%2522%253A%2522https%253A%252F%252Fwww.yuque.com%252Fattachments%252Fyuque%252F0%252F2019%252Fzip%252F216914%252F1573805361157-7fe55f8e-8acd-4db5-a065-78142b19d7bc.zip%2522%252C%2522card%2522%253A%2522file%2522%257D) 也可直接执行

`git clone[https://github.com/pjreddie/darknet](https://link.zhihu.com/?target=https%3A//github.com/pjreddie/darknet)`将源码下载到本地

### 权重文件

[yolov3-tiny.weights](https://link.zhihu.com/?target=https%3A//pjreddie.com/media/files/yolov3-tiny.weights)

[yolov2.weights](https://link.zhihu.com/?target=https%3A//pjreddie.com/media/files/yolov2.weights)

[yolov3.weights](https://link.zhihu.com/?target=https%3A//pjreddie.com/media/files/yolov3.weights)

[darknet53.conv.74](https://link.zhihu.com/?target=https%3A//pjreddie.com/media/files/darknet53.conv.74)

### VOC 数据集

[VOCtrainval_11-May-2012.tar](https://link.zhihu.com/?target=https%3A//pjreddie.com/media/files/VOCtrainval_11-May-2012.tar)

[VOCtrainval_06-Nov-2007.tar](https://link.zhihu.com/?target=https%3A//pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar)

[VOCtest_06-Nov-2007.tar](https://link.zhihu.com/?target=https%3A//pjreddie.com/media/files/VOCtest_06-Nov-2007.tar)

### 其他：

[YOLO-V3 可视化训练过程中的参数，绘制 loss、IOU、avg Recall 等的曲线图](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_34806812/article/details/81459982)

[AlexyAB 大神总结的优化经验](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/pprp/p/10204480.html)

2. 源码编译
-------

2.1 编辑 Makefile
---------------

**指定是否使用 GPU** 在执行 make 指令编译之前，需要编辑一下 makefile，来指定是否需要用 GPU(cuda)，如果用 cuda，是否需要用 cudnn 加速；是否需要用 opencv 等。我这里是用 GPU 且需要用 CUDNN 加速的，不使用 OpenCV。 Makefile 的前 5 行如下，这 5 项内容 0 表示不启用，1 表示启用，可根据需求自己配置。

*   **GPU 是否启用 GPU1**
*   **CUDNN 是否启用 CUDNN 加速，若 GPU = 1 则 CUDNN 可选 1 或 0；GPU=0 则 CUDNN=0**
*   **OPENCV 是否启用 OpenCV, 启用的话需先编译安装好，启用可支持对视频和图像流文件处理**
*   **OPENMP 是否启动多核 CPU 来加速 Yolo, 如果是用 CPU 训练，建议开启 = 1**
*   **DEBUG 表示编译的 Yolo 版本为是否为 DEBUG 版**

**![](https://pic3.zhimg.com/v2-32baf9d22762a3dd4371e7ad33d46b3e_r.jpg)**

如果不使用 GPU 则 GPU=0,CUDNN=0；还有一点需注意，如果在 shell 执行 nvcc 找不到命令（没有添加到环境变量），则需要将 nvcc 命令的全路径写出来 **指定 cuda 的路径** 如果不使用 GPU, 则此步可忽略，如果使用 GPU, 则需要指定 cuda 路径。官方的 Makefile 默认的 cuda 路径为：/usr/local/cuda，如果你安装了多个 cuda，或者 cuda 路径更改过，则需要指定你的 cuda 路径，这里需要改两处：51 行的 COMMON 和 53 行的 LDFAGS

![](https://pic3.zhimg.com/v2-3af8efefb47076b1ca5c510a936ed2b6_r.jpg)

2.2 执行 make
-----------

![](https://pic2.zhimg.com/v2-84edb71f58348b021b7b8bb7ab231bc1_r.jpg)

执行 make 编译完成后，在项目主目录下生成可执行的 darknet 文件。然后我们就可以用这个可执行文件来进行模型训练和测试了！

3. 功能测试
-------

将下载好的权重文件放入主目录下，然后 cd 到该目录下执行：

```
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

如果你看到如下输出，且在主目录下找到一张 predictions.jpg 的图片，则执行成功，表明上一步骤中编译的 darknet 可以正常使用。

![](https://pic4.zhimg.com/v2-92dc3c31ad244e733909229414f39ccf_r.jpg)

你也可以指定一个阈值，yolo 默认输出置信度 > 0.25 的预测框，你也可以自行指定：

```
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0.45
```

-thresh 0.45 表示：置信度低于 45% 的预测框都不会被输出。

除了使用经典的 yolov3 模型外，你还可以换一个模型尝试，譬如 yolov3-tiny.weights

```
./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg
```

当然，也可以通过连接摄像头实现实时视频画面预测。（需要编译时添加 opencv）

```
./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <video file>
```

4. 模型训练
-------

4.1 数据准备
--------

模型训练前首先准备数据集，我们用 VOC 数据集，将 VOC 数据集解压，解压后共同存放在 VOCevkit 文件夹中，我将 VOCevkit 放在了 darknet 主文件夹 / data/VOC / 下。 cd 到 / VOC 目录下，下载 [voc_label.py](https://link.zhihu.com/?target=https%3A//www.yuque.com/attachments/yuque/0/2019/py/216914/1573815884084-4f9be189-93ba-429d-9fe5-b67441e5843c.py%3F_lake_card%3D%257B%2522uid%2522%253A%25221573815883860-0%2522%252C%2522src%2522%253A%2522https%253A%252F%252Fwww.yuque.com%252Fattachments%252Fyuque%252F0%252F2019%252Fpy%252F216914%252F1573815884084-4f9be189-93ba-429d-9fe5-b67441e5843c.py%2522%252C%2522name%2522%253A%2522voc_label.py%2522%252C%2522size%2522%253A2241%252C%2522type%2522%253A%2522text%252Fx-python%2522%252C%2522ext%2522%253A%2522py%2522%252C%2522progress%2522%253A%257B%2522percent%2522%253A0%257D%252C%2522status%2522%253A%2522done%2522%252C%2522percent%2522%253A0%252C%2522id%2522%253A%2522zzTMN%2522%252C%2522card%2522%253A%2522file%2522%257D) ，并运行：

```
python voc_label.py
```

> 可以将该脚本复制到 / scripts 下留作备份

文件夹下会生成 7 个. txt 文件：

![](https://pic4.zhimg.com/v2-b7110876b44be5660793ca69ba3a29f3_r.jpg)

如果你的 voc_label.py 脚本是从官网 wget [https://pjreddie.com/media/files/voc_label.py](https://link.zhihu.com/?target=https%3A//pjreddie.com/media/files/voc_label.py) 下载的，则需要在脚本 57 行处额外加上如下两行内容：

```
os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
```

目的：将 2007 年的训练和验证图像 + 2012 年的图像都放入了 train.txt 用于集中训练

4.2 修改配置文件 voc.data
-------------------

配置 cfg/voc.data, 来确定你需要检测的目标类别数量和名称；修改 train 和 valid 的图片资源路径。训练资源指向 train.txt 测试 / 验证资源指向 2007_test.txt

![](https://pic4.zhimg.com/v2-4428bf5e3bde38da6c8df12b62621dd3_r.jpg)

VOC 数据集默认的类别数量为 20 个，名称在 data/voc.names 中：

![](https://pic1.zhimg.com/v2-0338a9035eea551ddc7b405919ef051c_b.png)

我这里使用默认的 20 个类别，所以无需修改；如果，你只想检测其中的几个类别，譬如 person 和 car，那么可以设置 voc.data 中 classes=2,names 只保留 person 和 car 这两种，不过后面相应的需要更改 yolov3-voc.cfg 里的卷积层配置等。

4.3 修改 yolov3-voc.cfg
---------------------

yolov3-voc.cfg 文件定义了 yolo 的卷积神经网络模型，和超参数配置等。这里需要注意，训练时需要将 #Testing 区块下的 batch 和 subvisions 注释掉；测试和验证时则放开注释，同时注释掉 #Training 区块下的内容。 训练时，可以根据自己 GPU 的内存来设置批的大小 batch，可设为 16、32、64、128 验证时，batch 和 subvisions 同时设为 1 即可。

![](https://pic3.zhimg.com/v2-52f8b15fc67c36ad3999aa6d1acc6826_b.jpg)

4.4 开始训练
--------

cd 回项目主目录，将 darknet53.conv.74 权重放入主目录下，运行：

```
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74
```

如果报错提示 cuda 内存溢出，可以降低 batch，再次运行；运行过程中可以通过 nvidia-smi 查看显存占用情况：

![](https://pic4.zhimg.com/v2-1492f47f95c8a51fa590966c07b1111f_r.jpg)

训练过程中会持续往 backup / 下生成权重文件，如：yolov3-voc_100.weights、yolov3-voc_200.weights....

5. 模型验证
-------

检验生成的模型权重文件、测试准确率和 map 等指标。验证前需要将 yolov3-voc.cfg 中的 batch 和 subdivisions 都改成 1。然后找到需要测试的权重文件，默认权重文件保存在：项目目录 / bakcup / 下，我选择这个 yolov3-voc_10000.weights，训练大约一个晚上 12 小时左右，loss 在 0.6 多。

![](https://pic1.zhimg.com/v2-d819e7b274025c8fcd8d995eb76e2fac_r.jpg)

5.1 测试
------

### 测试一

我们从 VOC 数据集中随便找一找图片来测试一下，这里我选取 000275.jpg 放入主目录下

![](https://pic2.zhimg.com/v2-d7dbe49af3568d26eca7a523ae89be0d_r.jpg)

运行：

```
./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_10000.weights 000275.jpg
```

![](https://pic2.zhimg.com/v2-49445b74041b988e26f48724a48f71fd_r.jpg)

**运行结束会在控制台打印出标注出的目标类别以及置信度、同时会在目录下生成一张标注图片：predictions.jpg** **

![](https://pic1.zhimg.com/v2-1955440ef06bfd93fc98723619044c98_r.jpg)

### 测试二

再随便找一张图片试试

```
./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_10000.weights data/person.jpg
```

![](https://pic2.zhimg.com/v2-e183d908e2c6fa62a3c7b1c972427cb9_r.jpg)![](https://pic4.zhimg.com/v2-c4730842d1913219822b235293eebb03_r.jpg)

5.2 验证
------

测试只能一张张地直观感受下模型的训练效果，看看标注出的目标是否正确，通过验证才能确定模型的整体训练效果，验证时会用模型对所有的 4952 张测试集图片进行预测，同时与正确结果进行比对，得出一个模型预测准确率。

### 验证测试集

可以运行以下脚本进行验证：

```
./darknet detector valid cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_10000.weights
```

验证完会在 results 文件夹下生成 20 个类别的验证结果 txt 文件

![](https://pic2.zhimg.com/v2-63578bf0a70ee3a6a147781906811e95_r.jpg)

默认的文件命名规则是：'comp4_det_test_' + '类别名称' + .txt，如 bird 类生成 comp4_det_test_bird.txt 文件。

![](https://pic3.zhimg.com/v2-97953b0f231727048ad58d71d1c839a6_r.jpg)

如图，第一列 000053 表示图像编号；第二列为置信度；后 4 列为检测出的目标框坐标

### 计算 map

计算 map, 我们需要两个 python 文件：

*   **voc_eval.py**
*   **compute_mAP.py**

其中 voc_eval.py 是 github 上开源项目的代码 [voc_eval.py](https://link.zhihu.com/?target=https%3A//github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py)；compute_mAP.py 需要我们自己编写

**voc_eval.py** 我们为了适配 darknet 中 voc 数据集的验证，需要对源码做几处修改： **a. 源码第 9 行：`import cPickle` 改为：**

```
import _pickle as cPickle
```

因为源码是 python2.x 版本，如果 python3.x 版本的运行会报错，故需要修改。

**b. 源码 103 行： `imagenames ``=`` [x.strip() ``for`` x ``in`` lines]`改为：**

```
imagenames = [x.strip().split('/')[-1].split('.')[0] for x in lines]
```

这个主要是为了方便适配 2007_test.txt，因为我们验证时采用的是 2007_test.txt 中的测试集图片，文件中存放的是图片的全路径，而 imagenames 需要用的是文件名、所以需要将全路径做个转换，截取到文件名。

**c.115 行`with`` ``open``(cachefile, ``'w'``) ``as`` f:`和 119 行`with`` ``open``(cachefile, ``'r'``) ``as f:`改成：**

```
with open(cachefile, 'wb') as f:
with open(cachefile, 'rb') as f:
```

如果不修改成‘wb’和‘rb’存储二进制格式，运行时会报错

**compute_mAP.py** 新建 compute_mAP.py 文件，用于调用 voc_eval.py, 内容如下:

```
from voc_eval import voc_eval
import os
map_ = 0
# classnames填写训练模型时定义的类别名称
classnames = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
for classname in classnames:
    ap = voc_eval('../results/{}.txt', '../data/VOC/VOCdevkit/VOC2007/Annotations/{}.xml', '../data/VOC/2007_test.txt', classname, '.')
    map_ += ap
    #print ('%-20s' % (classname + '_ap:')+'%s' % ap)
    print ('%s' % (classname + '_ap:')+'%s' % ap)
# 删除临时的dump文件
if(os.path.exists("annots.pkl")):
    os.remove("annots.pkl")
    print("cache file:annots.pkl has been removed!")
# 打印map
map = map_/len(classnames)
#print ('%-20s' % 'map:' + '%s' % map)
print ('map:%s' % map)
```

我这里在项目主目录下新建了一个【yolo-compute-util】文件夹，将两个. py 文件放在文件夹中，cd /yolo-compute-util，然后运行：`python compute_mAP.py`即可根据 results 中的验证结果统计出各个类别的 ap, 以及汇总的 map。

![](https://pic1.zhimg.com/v2-fb47858f5cfd6488660f421de5f9c548_r.jpg)

可以看见，经过 1 晚上的训练，我们的模型——yolov3-voc_10000.weights 的 mAP 是 0.740，虽然没有达到 yolov2 系列 76.8 + 的水平，不过一晚上的训练能达到如此程度，也是挺高了。

![](https://pic3.zhimg.com/v2-44b3bb5a12671caca926db56125d5f3a_r.jpg)