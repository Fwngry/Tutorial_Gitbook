> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/Creama_/article/details/106209388)

### 目录

*   [编译 darknet](#darknet_17)
*   [训练 PASCAL VOC2007 数据集](#PASCAL_VOC2007_44)
*   *   [准备预训练模型和数据集](#_46)
    *   [生成 darknet 需要的 label 文件](#darknetlabel_48)
    *   [修改几个配置文件](#_69)
    *   [训练](#_124)
    *   [段错误](#_146)
    *   [测试](#_154)
*   [训练自己的数据集](#_166)
*   *   [训练完 map 为 0 的问题](#map0_288)
*   [批量测试图片并保存](#_306)
*   [将 weights 权值文件转换为 tflite 权值文件](#weightstflite_361)
*   [附加内容](#_396)
*   *   [显示检测框的置信度](#_398)
    *   [改变检测框的粗细](#_399)
    *   [保存检测框的内容到本地 (批量图片检测)](#_400)
    *   [保存检测框的内容到本地 (视频检测)](#_401)

**如果希望先训练 PASCAL VOC 数据集，可以按顺序阅读，如果`想直接训练自己的数据集`，可以先看`编译darknet`部分， 然后直接跳到`训练自己的数据集`部分。**

yolov4 出来有一段时间了，我也用 yolov4 训练了自己的数据集，效果还是非常不错的。本文记录了用 yolov4 训练并测试 PASCAL VOC 和自己制作的数据集的方法，并且记录了如何将 weights 文件转换为移植到安卓端需要用到的 tflite 文件。  
yolov4 的 paper: [YOLOv4](https://arxiv.org/abs/2004.10934)  
yolov4 的代码地址: [darknet](https://github.com/AlexeyAB/darknet)

参考博文：[基于 Darknet 深度学习框架训练 YoloV4 模型，并用自己的模型批量处理图片并保存在文件夹内](https://blog.csdn.net/weixin_42770591/article/details/105863701)  
系统环境：  
GeForce GTX 1080 Ti  
CUDA 9.1  
Cudnn 7.1.3  
Python 3.6.7  
因为服务器和本机的图形界面交互没做好，所以没有用 OpenCV。  
（2020-7-9 更正：服务器和本机的图形交互用 xmanger 就好啦，然后编译时就可以令 OPENCV=1）

编译 darknet
==========

打开 darknet-master/Makefile，如果没有下载 darknet，命令行输入：

```
git clone https://github.com/AlexeyAB/darknet.git
```

在 darknet 下找到 Makefile 文件，并修改 Makefile：

```bash
cd ./darknet
ls
vi Makefile
```

Q:Linux中如何修改文件内容并保存?

1. 如果想修改这个文件的内容，按下间键盘上的'i'键，最下方就会变成INSERT，就可以修改了
2. 写个echo的代码进行保存 Esc+:+wq 这个是保存，回车保存，保存完成以后，再次打开这个文件会发现刚才写入的代码已经保存到里面了。
3. 不保存的命令Esc+:+q!这个是不保存

```
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=0
AVX=0
OPENMP=0
LIBSO=0
```

然后命令行输入：

```
cd darknet
make
```

编译完成后输入`./darknet`， 如果出现`usage:./darknet<fuction>`说明编译成功。  
可以用作者训练好的权值文件 [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) 测试一下 (下载速度太慢可私信邮箱)，下载后放在 darknet 目录下，在命令行输入：

```bash
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
./darknet detect cfg/yolov4.cfg yolov4.weights data/horses.jpg
```

因为没有用 OpenCV，所以检测结果没有直接显示，而是保存为 darknet/predictons.jpg，看一下检测效果：  
![](https://img-blog.csdnimg.cn/20200519162818378.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NyZWFtYV8=,size_16,color_FFFFFF,t_70)

训练 PASCAL VOC2007 数据集
=====================

作者在 github 上给的是训练 PASCAL VOC2007+2012 案例，因为我只下载了 VOC2007 的数据集，所以介绍只训练 VOC2007 的方法，其实方法都差不多。

准备预训练模型和数据集
-----------

首先要下载预训练模型 [yolov4.conv.137](https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view)(下载速度太慢可私信邮箱)，并放在 darknet 目录下，将 VOC2007 数据集放在 darknet/data/voc/VOCdevkit 目录下。

生成 darknet 需要的 label 文件
-----------------------

将 darknet/scripts 目录下的 voc_label.py 文件复制到 darknet/data/voc 目录下：`cp scripts/voc_label.py data/voc/`，并修改`voc_label.py`文件。  
将开头改为：

```
# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = [('2007', 'train'),('2007', 'val'),('2007', 'test')]
```

最后两行改为：

```
# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
os.system("cat 2007_train.txt 2007_val.txt > train.txt")
os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt > train.all.txt")
```

修改完成后运行：

```
cd data/voc
python voc_label.py
```

运行结果产生五个文件 2007_train.txt, 2007_test.txt, 2007_val.txt, train.all.txt, train.txt。  
![](https://img-blog.csdnimg.cn/20200519172832670.png)

修改几个配置文件
--------

**1.cfg/voc.data**

```
classes= 20
train  = /home/zmh/darknet/data/voc/train.txt		# 修改为训练集路径
valid  = /home/zmh/darknet/data/voc/2007_test.txt		# 修改为测试集路径
names = data/voc.names		# 类别名称文件
backup = backup		# 模型保存文件夹
```

**2.cfg/yolo-obj.cfg**  
将 cfg/yolov4-custom.cfg 复制，并将复制后的文件命名为 yolo-obj.cfg：

```
cd cfg
cp yolov4-custom.cfg yolo-obj.cfg
```

修改 yolo-obj.cfg:  
**第 1 处**

```
[net]
#Testing
#batch=1
#subdivisions=1
#Training
batch=64
subdivisions=16		# 如果训练时出现out of memory，可以修改为32或者64
width=608		# 修改为416
height=608		# 修改为416
channels=3
```

**第 2 处**

```
learning_rate=0.001
burn_in=1000
max_batches = 500500		# 修改为(类别数*2000)
policy=steps
steps=400000,450000		# 修改为0.8*max_batches，0.9*max_batches
scales=.1,.1
```

**第 3、4、5 处**  
ctrl+f 查找`yolo`，一共有三处，修改三处 yolo 上面的 filters，以及 yolo 下面的 classes：

```
[convolutional]
size=1
stride=1
pad=1
filters=255		# 修改为(类别数+5)*3
activation=linear


[yolo]
mask = 0,1,2
anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
classes=80		# 修改为类别数，20
num=9
```

训练
--

```
./darknet detector train cfg/voc.data cfg/yolo-obj.cfg yolov4.conv.137
```

训练得到的权值文件保存在 darknet/backup 中。  
训练是默认用第 0 块显卡，如果要指定第 x 块显卡，可以：

```
./darknet detector train cfg/voc.data cfg/yolo-obj.cfg yolov4.conv.137 -i x
```

如果要使用多 GPU 训练，可以先使用一个 GPU 训练得到 1000 次的权值文件 yolo-obj_1000.weights，然后停止训练，接着使用多 GPU 训练（最多四块）：

```
./darknet detector train cfg/voc.data cfg/yolo-obj.cfg yolov-obj_1000.weights -gpus 0,1,2,3
```

如果要保存 log 文件：

```
# tee后面是log文件名字，自己设置
./darknet detector train cfg/voc.data cfg/yolo-obj.cfg yolov4.conv.137 | tee yolo_train.log
```

在使用 OPENCV 的情况下，如果要画 mAp 曲线：

```
./darknet detector train cfg/voc.data cfg/yolo-obj.cfg yolov4.conv.137 -map
```

段错误
---

如果训练时出现段错误：  
mosaic=1 -compile Darknet with OpenCV for using mosaic=1  
free(): corrupted unsorted chunks  
![](https://img-blog.csdnimg.cn/20200519142112132.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NyZWFtYV8=,size_16,color_FFFFFF,t_70)  
作者有给解决方法：  
![](https://img-blog.csdnimg.cn/20200519142312697.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NyZWFtYV8=,size_16,color_FFFFFF,t_70)  
因为我没有用 OpenCV，所以要将 cfg/yolo-obj.cfg 中的`mosaic`设置为 0。

测试
--

测试图片：

```
./darknet detector test cfg/voc.data cfg/yolo-obj.cfg backup/yolo-obj_8000.weights data/person.jpg
```

测试视频（需要 OpenCV）：

```
./darknet detector demo cfg/voc.data cfg/yolo-obj.cfg backup/yolo-obj_8000.weights  test.mp4 -out_filename output.mp4
```

FPS 很小的话，加一个 - dont_show 或者最小化 xmanger 就行了。  
训练了 8000 次的检测效果：  
![](https://img-blog.csdnimg.cn/20200519142925556.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NyZWFtYV8=,size_16,color_FFFFFF,t_70)

训练自己的数据集
========

完成了 VOC2007 的训练和测试过程，离训练自己的数据集只有一步之遥了，为了方便处理，我把自己的数据集整理成 PASCAL VOC 的格式，具体过程见[轻松自制 PASCAL VOC 数据集](https://blog.csdn.net/Creama_/article/details/105974455)。  
假设自己的数据集已经被制作成了 PASCAL VOC 格式了，并且命名为 VOC 2007。如果还没有下载预训练模型 [yolov4.conv.137](https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view)(下载速度太慢可私信邮箱)，需要先下载，下载完成放在 darknet 目录下，并将自己的数据集 (VOC 2007) 放在 darknet/data/voc/VOCdevkit 目录下。接下来进行以下几个跟训练 PASCAL VOC 数据集类似的操作（我的数据集只有`两个`类别：green light 和 red light）：  
**1. 生成 darknet 需要的 label 文件**  
**如果之前已经修改过 voc_label.py 文件了**，重新`python voc_label.py`就行了，然后会生成五个新的文件：2007_train.txt, 2007_test.txt, 2007_val.txt, train.all.txt, train.txt。  
**如果之前没有训练过 PASCAL VOC 数据集**，将 darknet/scripts 目录下的 voc_label.py 文件复制到 darknet/data/voc 目录下：`cp scripts/voc_label.py data/voc/`，并修改`voc_label.py`文件。  
将开头改为：

```
# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = [('2007', 'train'),('2007', 'val'),('2007', 'test')]

classes = ["green light", "red light"]	# 修改为自己的类别
```

最后两行改为：

```
# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
os.system("cat 2007_train.txt 2007_val.txt > train.txt")
os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt > train.all.txt")
```

修改完成后运行：

```
cd data/voc
python voc_label.py
```

同样的，会生成五个新的文件：2007_train.txt, 2007_test.txt, 2007_val.txt, train.all.txt, train.txt。  
**2. 修改 voc.data 文件**  
如果训练过 PASCAL VOC，只要修改其中的 classes。否则根据注释修改：

```
classes= 2		# 改为自己数据集的类别数
train  = /home/zmh/darknet/data/voc/train.txt		# 修改为训练集路径
valid  = /home/zmh/darknet/data/voc/2007_test.txt		# 修改为测试集路径
names = data/voc.names		# 类别名称文件，下面会讲怎么修改里面的内容
backup = backup		# 模型保存文件夹
```

**3. 修改 cfg/yolo-obj.cfg 文件**  
如果没有训练过 PASCAL VOC，需要将 cfg/yolov4-custom.cfg 复制，并将复制后的文件命名为 yolo-obj.cfg：

```
cd cfg
cp yolov4-custom.cfg yolo-obj.cfg
```

然后修改 yolo-obj.cfg:  
**第 1 处，如果训练过 PASCAL VOC，可不用修改：**

```
[net]
#Testing
#batch=1
#subdivisions=1
#Training
batch=64
subdivisions=16		# 如果训练时出现out of memory，可以修改为32或者64
width=608		# 修改为416
height=608		# 修改为416
channels=3
```

**下面几处即使训练过 PASCAL VOC，同样要对应修改，第 2 处：**

```
learning_rate=0.001
burn_in=1000
max_batches = 4000		# 修改为(类别数*2000)
policy=steps
steps=3200,3600		# 修改为0.8*max_batches，0.9*max_batches
scales=.1,.1
```

**以及第 3-5 处：**  
ctrl+f 查找`yolo`，修改三处 yolo 上面的 filters，以及 yolo 下面的 classes：

```
[convolutional]
size=1
stride=1
pad=1
filters=21		# 修改为(类别数+5)*3
activation=linear


[yolo]
mask = 0,1,2
anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
classes=2		# 修改为类别数，20
num=9
```

**4. 修改 data/voc.names 文件**（`注意，之前训练PASCAL数据集是没有这个步骤的`）  
voc.names 文件中保存的是 VOC 数据集的类别（20 个），我们要修改为自己数据集的类别。

```
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```

改为：

```
green light
red light
```

**5. 训练**  
训练之前，转移或删除 darknet/backup 中之前得到的权值文件。

```
./darknet detector train cfg/voc.data cfg/yolo-obj.cfg yolov4.conv.137
```

如果要打印 loss 曲线，同时打印 map 值，在后面加 - map：

```
./darknet detector train cfg/voc.data cfg/yolo-obj.cfg yolov4.conv.137 -map
```

训练完得到四个权值文件（已重命名）：  
![](https://img-blog.csdnimg.cn/20200519151020693.png)  
指定 GPU 或者多 GPU 训练、打印训练 log、段错误的解决方法等，可以看训练 PASCAL VOC 数据集的训练部分。

训练完 map 为 0 的问题
---------------

如果训练完发现 map 为 0，有可能是：1. 没有改 voc_label.py 中的类名。2. 没有使用正确的预训练模型 (yolov4.conv.137)。3. 数据集有问题，或者没有成功转为 yolo 格式。  
如果要计算每个权值文件的 mAP 值，可以：

```
./darknet detector map cfg/voc.data cfg/yolo-obj.cfg backup/yolo-tflight_1000.weights
```

然后可以选一个 mAP 值最高的权值文件测试。![](https://img-blog.csdnimg.cn/20200519151234506.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NyZWFtYV8=,size_16,color_FFFFFF,t_70)  
**6. 测试**  
测试单张图片：

```
./darknet detector test cfg/voc.data cfg/yolo-obj.cfg backup/yolo-tflight_4000.weights
```

测试视频 (需要 OpenCV)：

```
./darknet detector demo cfg/voc.data cfg/yolo-obj.cfg backup/yolo-tflight_4000.weights -ext_output test.mp4
```

看一下效果：  
![](https://img-blog.csdnimg.cn/20200718112306984.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NyZWFtYV8=,size_16,color_FFFFFF,t_70)

批量测试图片并保存
=========

如果要测试大量图片，一张一张测试是非常麻烦的，为了能够批量测试图片，并保存在指定文件夹中，需要以下操作：  
**1. 修改 darknet/src/detector.c 文件**  
在 detector.c 文件的开头位置，添加`GetFilename`函数，为了解决上个版本文件名长度的问题，这里做了修改，直接将上个版本的常数用变量 length 替代，length 是测试图片的文件名，因为要去掉 ".jpg"，所以 strncpy 那里要用 "`length-4`"。代码亲测有效。

```
static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

// 在下面添加GetFilename函数
char *GetFilename(char *p)
{ 
    static char name[64];	// 作为返回值不能是局部变量
    memset(name, 0, sizeof(name));	// 清空之前的静态变量
    int length = 0;
    
    char *q = strrchr(p,'/') + 1;	// 获取第一个'/'后面的位置
    length = strlen(q);
    strncpy(name,q,length-4);	// 赋值并且不要后缀
    return name;
}
```

ctrl+f 查找`save_image`，查找下一个，然后找到 "`save_image(im, "predictions")`"，替换源代码：

```
// save_image(im, "predictions");
char b[64];
snprintf(b, 64, "output/%s", GetFilename(input));
save_image(im, b);
```

然后在 darknet 目录下重新`make`。  
**2. 准备包含所有需要检测图片路径的 txt 文件**  
将需要检测的图片放在 darknet/data/test 文件夹下，然后在 darknet/data 下新建一个 py 文件：

```
touch make_test_txt.py
```

文件中的内容为：

```
# coding: utf-8
import os

paths = "./test" 	# 测试图片的路径
f = open('test.txt', 'w')  	# 最后得到的图片路径txt文件
filenames = os.listdir(paths)
filenames.sort()
for filename in filenames:
    out_path = "data/test/" + filename 	# 引号内为测试图片文件夹的路径
    print(out_path)
    f.write(out_path + '\n')
f.close()
```

然后运行`make_test_txt.py`，在 data 目录下会生成一个`test.txt`文件。  
**3. 批量测试**  
在 darknet 目录下新建一个文件夹`output`，运行：

```
./darknet detector test cfg/voc.data cfg/yolo-obj.cfg backup/yolo-tflight_4000.weights -ext_output -dont_show <data/test.txt> result.txt
```

`注意，保存测试图片文件名的test.txt文件两边有< >`，result.txt 是把检测的文本输出打印出来，可有可无，检测的图片输出保存在`output`文件夹中。

将 weights 权值文件转换为 tflite 权值文件
=============================

将模型移植到安卓端需要将 weights 文件转换为 tflite 文件，系统环境：  
Tensorflow 2.1.0  
tensorflow_addons 0.9.1  
运行时可能会提升缺少其他安装包，根据提示使用 pip 安装即可。  
首先克隆代码到本地：

```
git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git
```

然后只需要修改`tensorflow-yolov4-tflite-master/data/classes/coco.names`，将内容修改为自己的类别名：

```
green light
red light
```

**或者**在 data/classes 中新建一个. names 文件，比如 “tfclight.names”，内容为自己的类名。然后打开 core/config.py，将

```
__C.YOLO.CLASSES              = "./data/classes/coco.names"
```

改为

```
__C.YOLO.CLASSES              = "./data/classes/tfclight.names"
```

接着将要转换的 weight 文件放在 data 目录下。  
最后在 tensorflow-yolov4-tflite-master 目录下，选择以下其中一条命令运行：

```
# yolov4非量化转换
python convert_tflite.py --weights ./data/yolo-tflight_4000.weights --output ./data/yolo-tflight_4000.tflite

# yolov4 quantize float16（量化转换）
python convert_tflite.py --weights ./data/yolo-tflight_4000.weights --output ./data/yolo-tflight_4000-fp16.tflite --quantize_mode float16
```

得到的结果：  
![](https://img-blog.csdnimg.cn/20200519170535427.png)  
参考：[tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)

附加内容
====

附加的内容由于篇幅原因，请移步到[另一篇博客](https://blog.csdn.net/Creama_/article/details/107960513)：https://blog.csdn.net/Creama_/article/details/107960513

显示检测框的置信度
---------

改变检测框的粗细
--------

保存检测框的内容到本地 (批量图片检测)
--------------------

保存检测框的内容到本地 (视频检测)
------------------