> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/Delusional/article/details/108974761?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242)

### 一、通过 Xshell 等工具连接到 Linux 服务器

由于我使用的是 window 电脑，所以需要借助第三方工具来连接服务器，使用 mac 系统或者 Linux 系统的话，可以直接在终端使用 ssh 连接到远程服务器。

可以在服务器中新建一个文件夹来放置项目所需的各种文件。

### 二、检查服务器中的 cuda 版本

在服务器终端中输入以下命令，查看 cuda 的版本，注意 V 需要大写

```
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar 
tar xvf VOCtrainval_11-May-2012.tar
```

可以看到服务器中的 cuda 版本是 10.0

![](https://img-blog.csdnimg.cn/20201009105900966.png)

之所以要检查 cuda 的版本，是因为之前在 cuda9.0 版本下，训练数据集后，最后的检测失败了（但也不能排除我操作上的原因），看了许多博客，如果最后的检测不出结果，有可能是 cuda 版本的原因，这一点需要格外注意一下。

下载安装新的 cuda，可以查询相关的博客。

### 三、下载安装 darknet

执行命令：

```
git clone https://github.com/pjreddie/darknet.git
```

执行成功后会在当前目录下自动生成一个 darknet 文件夹

![](https://img-blog.csdnimg.cn/20201009111904943.png)

使用命令进入 darknet：

```
cd darknet
```

文件夹中的文件如下，有些文件是后续生成或者上传进去的，所以你执行后的命令肯定跟下面截图的不一样，后续我会对这些多出来的文件一一讲解

![](https://img-blog.csdnimg.cn/20201009112424437.png)

### 四、更改配置文件 Makefile

使用 vim 编译器打开 Makefile 文档：

```
vim Makefile
```

将 GPU 和 CUDNN 后面的数值 0 改成 1，否则默认用 CPU 训练数据的话，速度会无比的缓慢，如果有安装 OpenCV，那么可以在第三行将 OPENCV 的值也置为一，这里我只是检测了图片，因此没有安装 OpenCV。

![](https://img-blog.csdnimg.cn/20201009112949879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RlbHVzaW9uYWw=,size_16,color_FFFFFF,t_70)

修改完成之后，**在命令行输入 make，否则修改的代码不会生效**

```
make
```

### 五、测试一下 yolov3 的效果

去 yolo 官网下载一个 yolov3.weights 的权重，下载命令如下：

```
wget https://pjreddie.com/media/files/yolov3.weights
```

下载过程可能会特别缓慢，或者是下载失败，这时候可以去找一些网盘的资源，下载到本地后，输入命令 rz 上传文件到 darknet 目录下，如图所示：

![](https://img-blog.csdnimg.cn/20201009122451894.png)

上传完成权重文件之后，就可以测试一下 yolov3 了

**确保当前目录在 darknet 下**，执行命令：

```
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

./darknet：当前目录

test：表示是检测，不是训练

cfg/coco.data：配置文件，包括物体总的类别数量，训练和测试文件的路径，生成权重保存在什么目录下等配置信息，在使用 yolov3 训练自己的数据集的时候是需要创建一个这样的文件的

![](https://img-blog.csdnimg.cn/20201009123635763.png)

cfg/yolov3.cfg：训练和测试过程的配置文件，训练 voc 数据集时，主要是要根据是训练还是测试，打开 / 关闭相应的注释，后面会再提到这个文件。

![](https://img-blog.csdnimg.cn/20201009123907104.png)

yolov3.weights：下载的权重文件，相当于就是已经训练的很好的权重信息，测试图片时可以直接拿来用

data/dog.jpg：要测试的图片，在 darknet/data 目录下，有一些其他的图片也可以拿来测试。

执行命令以后，会出现如下的结果：

![](https://img-blog.csdnimg.cn/20201009124257614.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RlbHVzaW9uYWw=,size_16,color_FFFFFF,t_70)

可以看到图片中的三个物体都被检测出来了

由于 Linux 服务器下不能查看图片，所以可以用 sz 命令，上传到本地来查看图片

![](https://img-blog.csdnimg.cn/20201009124527407.png)

上述的 predictions.jpg，就是已经标记好的图片，输入命令：

```
sz predictions.jpg
```

可以在本地查看到图片如下：

![](https://img-blog.csdnimg.cn/20201009124716309.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RlbHVzaW9uYWw=,size_16,color_FFFFFF,t_70)

### 六、下载上传 VOC 数据集

下载 VOC 数据集，官网如下，如果下载缓慢，也可以去找网盘资源

[https://pjreddie.com/projects/pascal-voc-dataset-mirror/](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

下载如下三个压缩包到本地，使用 rz 命令上传到服务器中，放到 darknet 目录下：

![](https://img-blog.csdnimg.cn/2020100912495058.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RlbHVzaW9uYWw=,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/2020100912513882.png)

### 七、解包生成相应的文件夹

使用命令进行解包：

```
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar 
tar xvf VOCtrainval_11-May-2012.tar
```

操作后，会在当前目录生成一个 VOCdevkit 文件夹，里面有 VOC2007，VOC2012 两个子文件夹，拿 VOC2012 文件夹举例，相应的目录结构如下：

![](https://img-blog.csdnimg.cn/20201009145433244.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RlbHVzaW9uYWw=,size_16,color_FFFFFF,t_70)

**这些文件夹解压时会自动生成，但是训练自己的 VOC 格式的数据集时需要手动创建**

### **八、执行 voc_label.py 文件**

将 scripts 文件下的 voc_label.py 拷贝到 darknet 目录下

```
cp scripts/voc_label.py ./
```

在 darknet 目录下，运行该 py 文件

```
python voc_label.py
```

运行完成后在 VOCdevkit/VOC2007（VOC2012）就可以看到多了一个 labels 文件![](https://img-blog.csdnimg.cn/20201009150422579.png)

在 darknet 目录下也会生成一些 txt 文件：

![](https://img-blog.csdnimg.cn/20201009150538948.png)

### 九、修改配置文件 voc.data

修改配置文件 voc.data。该文件位于 darknet/cfg 目录下，**修改 train 和 valid 后面的路径，改为自己的路径**

![](https://img-blog.csdnimg.cn/2020100915100930.png)

说明：

classes=20：VOC 数据集一共 20 个类别

train=...：训练文件的路径，train.txt 是运行上面的 py 文件生成的，每行都是训练图片的绝对路径

vaild=...：测试文件的路径，也是运行上面的 py 文件生成的

names=...：20 个类别的名称，voc.name 文件是内置在 darknet 里面的，训练 VOC 数据集时不用手动修改，但是训练自己的数据集时就要手动修改了

backup=...：训练过程中存放权重文件，有过程中的权重文件和最终训练完成的权重的文件

### 十、修改配置文件 yolov3-voc.cfg

该文件在 darknet/cfg 目录下，打开 yolov3-voc.cfg 文件，将第 6、7 行 training 下的注释去掉，如果按照默认的 batch=1，subdivisions=1，训练时会出现大量的 nan

![](https://img-blog.csdnimg.cn/20201009151829313.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RlbHVzaW9uYWw=,size_16,color_FFFFFF,t_70)

把里面的 max_batches 调小，可以显著降低训练时间

![](https://img-blog.csdnimg.cn/2020100915480288.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RlbHVzaW9uYWw=,size_16,color_FFFFFF,t_70)

### 十一、下载预训练权重并进行训练

终端下执行如下命令，下载预训练权重，注意是在 darknet 目录下执行该命令。如果速度慢，就找网盘资源。

```
wget https://pjreddie.com/media/files/darknet53.conv.74
```

完成上面的所有工作后，就可以进行训练了，训练的代码：

```
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74
```

train：代表训练

其他的参数在测试 yolov3 效果时说明过，总之就是指定一些配置文件和权重等

训练的时长在 GPU 性能较好的情况下大概需要好几个小时，但是训练一段时间，当损失降的比较低时，可以提前终止训练。

训练结束后，在 backup 文件夹中，可以看到许多过程的权重文件，里面的任何一个都可以用于测试，当然训练的越久，效果也就越好，**训练完成后，会有一个后缀为 final 的文件，那个就是最终的权重文件了（我这里偷懒没有训练到最后），注意 yolov3-voc.backup 也是可以用的，这个文件保存的是最新的权重信息。**

![](https://img-blog.csdnimg.cn/20201009152629687.png)

### 十二、测试

我测试的是单张图片，我是将图片上传到 data 文件夹下，下面的 XXX 是图片的名称，当然 data 文件夹里面有些本来就有的图片，那个也可以用来测试

```
./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_600.weights data/XXX.jpg
```

测试成功的话，会出现图片中某个物体的概率，（默认的阈值好像是 50%），**结果图片（也就是物体框起来的图片）就是 predictions.jpg**，上传到自己电脑就可以查看了。

测试时应该要修改配置文件 yolov3-voc.cfg（打开，关闭注释即可），但是经过测试，发现用训练的 batch 和 subdivisions 也是可以测试成功的

### 十三、测试过程指定置信度阈值

测试过程中可以手动指定置信度阈值，在测试代码后加上，数值可以随意指定

```
-thresh 0.25
```

**最后这个 0.25 就是置信度，代表相似度为 0.25 及以上的目标都会被标出**

### **十四、最后**

如果需要测试多张图片，可以参考：

**[YOLOv3 批量测试图片并保存在自定义文件夹下](https://blog.csdn.net/mieleizhi0522/article/details/79989754)**

如果想要了解一些参数的效果，可以参考：

**[Yolov3 参数理解](https://blog.csdn.net/weixin_42731241/article/details/81474920)**

**参考博客：**

**[【学习笔记—Yolov3】Yolov3 训练 VOC 数据集 & 训练自己的数据集](https://blog.csdn.net/weixin_43962659/article/details/86364660)**

**[Yolov3 参数理解](https://blog.csdn.net/weixin_42731241/article/details/81474920)**

[YOLOV3 训练自己的数据集（VOC 数据集格式）](https://blog.csdn.net/weixin_43818251/article/details/89548583?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-5.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-5.channel_param)

**[YOLO V3 置信度阈值调整](https://blog.csdn.net/r12345q__/article/details/90694068?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v25-1-90694068.nonecase&utm_term=yolov3%20%E9%98%88%E5%80%BC%E8%AE%BE%E5%AE%9A)**

**[YOLOv3 批量测试图片并保存在自定义文件夹下](https://blog.csdn.net/mieleizhi0522/article/details/79989754)**

**[YoLov3 训练自己的数据集（小白手册）](https://blog.csdn.net/weixin_42731241/article/details/81352013)**