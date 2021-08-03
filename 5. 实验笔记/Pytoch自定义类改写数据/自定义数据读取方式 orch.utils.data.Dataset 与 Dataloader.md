> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/28200166)

0. 本章内容
-------

在 pytorch 中，提供了一种十分方便的数据读取机制，即使用 torch.utils.data.Dataset 与 Dataloader 组合得到数据迭代器。在每次训练时，利用这个迭代器输出每一个 batch 数据，并能在输出时对数据进行相应的预处理或数据增广操作。

同时，pytorch 可视觉包 torchvision 中，继承 torch.utils.data.Dataset，预定义了许多常用的数据集，并提供了许多常用的数据增广函数。

本章主要进行下列介绍：

*   torch.utils.data.Dataset 与 Dataloader 的理解
*   torchvision 中的 datasets
*   torchvision ImageFolder
*   torchvision transforms

具体代码可以在 [XavierLinNow/pytorch_note_CN](https://link.zhihu.com/?target=https%3A//github.com/XavierLinNow/pytorch_note_CN) 得到

1. torch.utils.data.Dataset 与 torch.utils.data.DataLoader 的理解
-------------------------------------------------------------

1.  pytorch 提供了一个数据读取的方法，其由两个类构成：torch.utils.data.Dataset 和 DataLoader
2.  我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中
3.  torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式
4.  torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程迭代器输出数据集

下面我们分别介绍下 torch.utils.data.Dataset 以及 DataLoader

**1.1 torch.utils.data.Dataset**

1.  要自定义自己的 Dataset 类，至少要重载两个方法，__len__, __getitem__
2.  __len__返回的是数据集的大小
3.  __getitem__实现索引数据集中的某一个数据
4.  除了这两个基本功能，还可以在__getitem__时对数据进行预处理，或者是直接在硬盘中读取数据，对于超大的数据集还可以使用 lmdb 来读取

**下面将简单实现一个返回 torch.Tensor 类型的数据集**

```
from torch.utils.data import DataLoader, Dataset
import torch

class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print 'tensor_data[0]: ', tensor_dataset[0]

'''
输出
tensor_data[0]:  (
 0.6804
-1.2515
 1.6084
[torch.FloatTensor of size 3]
, 0.2058754563331604)
'''

# 可返回数据len
print 'len os tensor_dataset: ', len(tensor_dataset)

'''
输出：
len os tensor_dataset:  4
'''
```

**1.2 torch.utils.data.Dataloader**

1.  Dataloader 将 Dataset 或其子类封装成一个迭代器
2.  这个迭代器可以迭代输出 Dataset 的内容
3.  同时可以实现多进程、shuffle、不同采样策略，数据校对等等处理过程

```
tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print 'one batch tensor data: ', iter(tensor_dataloader).next()
# 输出batch数量
print 'len of batchtensor: ', len(list(iter(tensor_dataloader)))

'''
输出：
one batch tensor data:  [
 0.6804 -1.2515  1.6084
-0.1156 -1.1552  0.1866
[torch.FloatTensor of size 2x3]
, 
 0.2059
 0.6452
[torch.DoubleTensor of size 2]
]
len of batchtensor:  2
'''
```

2. torchvision.datasets
-----------------------

1.  pytorch 专门针对视觉实现了一个 torchvision 包，里面包括了许多常用的 CNN 模型以及一些数据集
2.  torchvision.datasets 包含了 MNIST，cifar10 等数据集，他们都是通过继承上述 Dataset 类实现的

**2.1 调用 torchvision 自带的 cifar10 数据集**

```
import torchvision.datasets as dset
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
%matplotlib inline

def imshow(img, is_unnormlize=False):
    if is_unnormlize:
        img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 载入cifar数据集
trainset = dset.CIFAR10(root='../data',                  # 数据集路径
                        train=True,                      # 载入train set
                        download=True,                   # 如果未下载数据集，则自动下载。
                                                         # 建议直接下载后压缩到root的路径
                        transform=transforms.ToTensor()  # 转换成Tensor才能被封装为DataLoader
                        )

# 封装成loader
trainloader = DataLoader(trainset, batch_size=4,
                         shuffle=True, num_workers=2)

# 显示图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
```

**显示图片**

![](https://pic2.zhimg.com/v2-e91ca3ebc5b343322c6ab86165fd5e01_b.png)

**2.2 直接从硬盘中载入自己的图像**

torch.datasets 包中的 ImageFolder 支持我们直接从硬盘中按照固定路径格式载入每张数据，其格式如下：

*   根目录 / 类别 / 图像

```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

3. torchvision.transforms
-------------------------

1.  在刚才，我们见到生成 cifar 数据集时有一个参数 transform，这个参数就是实现各种预处理
2.  在 torchvision.transforms 中，有多种预测方式，如 scale，centercrop
3.  我们可以使用 Compose 将这些预处理方式组成 transforms list，对图像进行多种处理
4.  需要注意，由于这些 transform 是基于 PIL 的，因此 Compose 中，Scale 等预处理需要先调用，ToTensor 需要后与他们
5.  如果觉得 torchvision 自带的预处理不够多，可以使用 [https://github.com/ncullen93/torchsample](https://link.zhihu.com/?target=https%3A//github.com/ncullen93/torchsample) 中的 transforms

```
# 定义transform
transform = torchvision.transforms.Compose(
    [transforms.RandomCrop(20),
     transforms.ToTensor(),                       # ToTensor需要在预处理之后进行
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )

# 载入cifar数据集
trainset = dset.CIFAR10(root='../data',                  # 数据集路径
                        train=True,                      # 载入train set
                        download=True,                   # 如果未下载数据集，则自动下载。
                                                         # 建议直接下载后压缩到root的路径
                        transform=transform              # 进行预处理
                        )

# 封装成loader
trainloader = DataLoader(trainset, batch_size=4,
                         shuffle=True, num_workers=2)

# 显示图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images), True)
```

**经过数据增广的数据：**

![](https://pic2.zhimg.com/v2-8656757e82a2627ff37e4f9ed8db81f9_b.png)