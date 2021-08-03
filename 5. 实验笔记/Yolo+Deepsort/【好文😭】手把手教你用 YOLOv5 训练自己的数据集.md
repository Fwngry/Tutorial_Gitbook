>  原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_44936889/article/details/110661862)

### 手把手教你用 YOLOv5 训练自己的数据集

1. 安装环境：Anaconda、CUDA、Pytorch、pip install -r requirements.txt、git clone repository
2. 数据准备：labelme打标签、spilt.py+txt2yolo_label.py 转换成yolo格式
3. 模型配置：data/* .yaml、model/* .yaml、weights/* .pt
4. 开始训练




![](https://img-blog.csdnimg.cn/20201204171203185.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

1. 安装 Anaconda
===============

Anaconda 官网：[https://www.anaconda.com/](https://www.anaconda.com/)

2. 创建虚拟环境
==========

这里我们需要为 yolov5 单独创建一个环境，输入：

```
conda create -n torch107 python=3.7
```


安装完成后，输入：

```
activate torch107
```

激活环境

3. 安装 pytorch
==============

yolov5 最新版本需要 pytorch1.6 版本以上，因此我们安装 pytorch1.7 版本。由于我事先安装好了 CUDA10.1，因此在环境中输入：

```
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

即可安装 
然后查看 CUDA 是否可用：  ![](https://img-blog.csdnimg.cn/20201204171539636.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

这里显示 True 表明正常安装。

4. 下载源码和安装依赖库
==============

源码地址：[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) 
![](https://img-blog.csdnimg.cn/20201204171644540.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)  安装依赖库：

```
pip install -r requirements.txt
```

![](https://img-blog.csdnimg.cn/20201204171807501.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

5. 数据标注
========

数据标注我们要用 labelimg，使用 pip 即可安装：

```
pip install labelimg
```

![](https://img-blog.csdnimg.cn/20201204172236887.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

5. 数据预处理
=========

创建 split.py 文件，内容如下：

```
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', default='VOCData/Annotations', type=str, help='input xml label path')
parser.add_argument('--txt_path', default='VOCData/labels', type=str, help='output txt label path')
opt = parser.parse_args()

trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
```

运行结束后，可以看到 VOCData/labels 下生成了几个 txt 文件：

![](https://img-blog.csdnimg.cn/20201204174635361.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)  
然后新建 txt2yolo_label.py 文件用于将数据集转换到 yolo 数据集格式：

```
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
from os import getcwd

sets = ['train', 'val', 'test']
classes = ['face', 'normal', 'phone', 'write',
           'smoke', 'eat', 'computer', 'sleep']


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    # try:
        in_file = open('VOCData/Annotations/%s.xml' % (image_id), encoding='utf-8')
        out_file = open('VOCData/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')
    # except Exception as e:
    #     print(e, image_id)


wd = getcwd()
for image_set in sets:
    if not os.path.exists('VOCData/labels/'):
        os.makedirs('VOCData/labels/')
    image_ids = open('VOCData/labels/%s.txt' %
                     (image_set)).read().strip().split()
    list_file = open('VOCData/%s.txt' % (image_set), 'w')
    for image_id in tqdm(image_ids):
        list_file.write('VOCData/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
```

转换后可以看到 VOCData/labels 下生成了每个图的 txt 文件：  ![](https://img-blog.csdnimg.cn/20201204180955419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

在 data 文件夹下创建 myvoc.yaml 文件：  ![](https://img-blog.csdnimg.cn/20201204173252942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)  
内容如下：

```
train: VOCData/train.txt
val: VOCData/val.txt

# number of classes
nc: 8

# class names
names: ["face", "normal", "phone", "write", "smoke", "eat", "computer", "sleep"]
```

6. 下载预训练模型
===========

我训练 yolov5m 这个模型，因此将它的预训练模型下载到 weights 文件夹下：  ![](https://img-blog.csdnimg.cn/20201204180345691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20201204180212858.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

7. 开始训练
========

修改 models/yolov5m.yaml 下的类别数：  ![](https://img-blog.csdnimg.cn/20201204180646283.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)  然后在 cmd 中输入：

```
python train.py --img 640 --batch 4 --epoch 300 --data ./data/myvoc.yaml --cfg ./models/yolov5m.yaml --weights weights/yolov5m.pt --workers 0
```

即可开始训练：  ![](https://img-blog.csdnimg.cn/20201204192019418.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

