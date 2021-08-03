> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/35698470)

之前刚开始用的时候，写 Dataloader 遇到不少坑。网上有一些教程 分为 all images in one folder 和 each class one folder。后面的那种写的人比较多，我写一下前面的这种，程式化的东西，每次不同的任务改几个参数就好。

等训练的时候写一篇文章把 2333

一. 已有的东西

举例子：用 kaggle 上的一个 dog breed 的数据集为例。数据文件夹里面有三个子目录

test: 几千张图片，没有标签，测试集

train: 10222 张狗的图片，全是 jpg，大小不一，有长有宽，基本都在 400×300 以上

labels.csv ： excel 表格, 图片名称 + 品种名称

![](https://pic4.zhimg.com/v2-6128d817c09f05fe3bdfe05e1f84a92f_r.jpg)

我喜欢先用 pandas 把表格信息读出来看一看

```
import pandas as pd
import numpy as np
df = pd.read_csv('./dog_breed/labels.csv')
print(df.info())
print(df.head())
```

![](https://pic1.zhimg.com/v2-9d680235e5ff00c3f869a3eab4630ca4_r.jpg)

看到，一共有 10222 个数据，id 对应的是图片的名字，但是没有后缀 .jpg。 breed 对应的是犬种。

二. 预处理

我们要做的事情是：

1) 得到一个长 list1 : 里面是每张图片的路径

2) 另外一个长 list2: 里面是每张图片对应的标签（整数），顺序要和 list1 对应。

3) 把这两个 list 切分出来一部分作为验证集

1）看看一共多少个 breed, 把每种 breed 名称和一个数字编号对应起来：

```
from pandas import Series,DataFrame

breed = df['breed']
breed_np = Series.as_matrix(breed)
print(type(breed_np) )
print(breed_np.shape)   #(10222,)

#看一下一共多少不同种类
breed_set = set(breed_np)
print(len(breed_set))   #120

#构建一个编号与名称对应的字典，以后输出的数字要变成名字的时候用：
breed_120_list = list(breed_set)
dic = {}
for i in range(120):
    dic[  breed_120_list[i]   ] = i
```

2）处理 id 那一列，分割成两段：

```
file =  Series.as_matrix(df["id"])
print(file.shape)

import os
file = [i+".jpg" for i in file]
file = [os.path.join("./dog_breed/train",i) for i in file ]
file_train = file[:8000]
file_test = file[8000:]
print(file_train)

np.save( "file_train.npy" ,file_train )
np.save( "file_test.npy" ,file_test )
```

里面就是图片的路径了

![](https://pic3.zhimg.com/v2-b740e480301df1fded91c92090065736_r.jpg)

3）处理 breed 那一列，分成两段：

```
breed = Series.as_matrix(df["breed"])
print(breed.shape)
number = []
for i in range(10222):
    number.append(  dic[ breed[i] ]  )
number = np.array(number) 
number_train = number[:8000]
number_test = number[8000:]
np.save( "number_train.npy" ,number_train )
np.save( "number_test.npy" ,number_test )
```

三. Dataloader

我们已经有了图片路径的 list,target 编号的 list。填到 Dataset 类里面就行了。

```
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor

#当然出来的时候已经全都变成了tensor
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        self.images = file_train
        self.target = number_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)
```

我们看一下代码，自定义 Dataset 只需要最下面一个 class, 继承自 Dataset 类。有三个私有函数

def __init__(self, loader=default_loader):

这个里面一般要初始化一个 loader(代码见上面), 一个 images_path 的列表，一个 target 的列表

def __getitem__(self, index)：

这里吗就是在给你一个 index 的时候，你返回一个图片的 tensor 和 target 的 tensor, 使用了 loader 方法，经过 归一化，剪裁，类型转化，从图像变成 tensor

def __len__(self):

return 你所有数据的个数

这三个综合起来看呢，其实就是你告诉它你所有数据的长度，它每次给你返回一个 shuffle 过的 index, 以这个方式遍历数据集，通过 __getitem__(self, index) 返回一组你要的（input,target）

四. 使用

实例化一个 dataset, 然后用 Dataloader 包起来

```
train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=4,shuffle=True)
```

![](https://pic1.zhimg.com/v2-8bbf753ce61d9a2cf0082b003b67d03c_r.jpg)