> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/wsp_1138886114/article/details/83620869)

### 文章目录

*   *   *   [一、torchvision 图像数据读取 [0, 1]](#torchvision__0_1_9)
        *   [二、torchvision 的 Transform](#torchvision__Transform_84)
        *   [三、读取图像数据类](#_133)
        *   *   *   [3.1 class torchvision.datasets.ImageFolder 默认读取图像数据方法：](#31_class_torchvisiondatasetsImageFolder__134)
                *   [3.2 自定义数据读取方法](#32__221)

运行环境安装 Anaconda | python ==3.6.6

```
conda install pytorch -c pytorch
pip install config
pip install tqdm             #包装迭代器，显示进度条
pip install torchvision
pip install scikit-image
```

### 一、torchvision 图像数据读取 [0, 1]

`import torchvision.transforms as transforms`  
transforms 模块提供了一般的图像转换操作类。  
**`class torchvision.transforms.ToTensor`**  
功能：  
把 shape=(H x W x C) 的像素值为 [0, 255] 的 PIL.Image 和 numpy.ndarray  
转换成 shape=(C x H x W) 的像素值范围为`[0.0, 1.0]`的 torch.FloatTensor。

**`class torchvision.transforms.Normalize(mean, std)`**  
功能：  
此转换类作用于 torch.*Tensor。给定均值 (R, G, B) 和标准差(R, G, B)，用公式 channel = (channel - mean) / std 进行规范化。

```
import torchvision 
import torchvision.transforms as transforms 
import cv2 
import numpy as np 
from PIL import Image 

img_path = "./data/timg.jpg" 

# 引入transforms.ToTensor()功能： range [0, 255] -> [0.0,1.0] 
transform1 = transforms.Compose([transforms.ToTensor()])

# 直接读取：numpy.ndarray 
img = cv2.imread(img_path)
print("img = ", img[0])      #只输出其中一个通道
print("img.shape = ", img.shape)

# 归一化，转化为numpy.ndarray并显示
img1 = transform1(img) 
img2 = img1.numpy()*255 
img2 = img2.astype('uint8') 
img2 = np.transpose(img2 , (1,2,0)) 
 
print("img1 = ", img1)
cv2.imshow('img2 ', img2 ) 
cv2.waitKey() 


# PIL 读取图像
img = Image.open(img_path).convert('RGB') # 读取图像 
img2 = transform1(img) # 归一化到 [0.0,1.0] 
print("img2 = ",img2) #转化为PILImage并显示 
img_2 = transforms.ToPILImage()(img2).convert('RGB') 
print("img_2 = ",img_2) 
img_2.show()


从上到下依次输出：---------------------------------------------
img =   [[197 203 202]
	 [195 203 202]
	 ...
	 [200 208 207]
	 [200 208 207]]
img.shape =  (362, 434, 3)

img1 =  tensor([[[0.7725, 0.7647, 0.7686,  ..., 0.7804, 0.7843, 0.7843],
         [0.7765, 0.7725, 0.7686,  ..., 0.7686, 0.7608, 0.7569],
         [0.7843, 0.7725, 0.7686,  ..., 0.7725, 0.7686, 0.7569],
         ...,

img_transform =  tensor([[[0.7922, 0.7922, 0.7961,  ..., 0.8078, 0.8118, 0.8118],
         [0.7961, 0.8000, 0.7961,  ..., 0.7922, 0.7882, 0.7843],
         [0.8039, 0.8000, 0.7961,  ..., 0.8118, 0.8039, 0.7922],
         ...,
```

![](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/05_17_watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzcF8xMTM4ODg2MTE0,size_16,color_FFFFFF,t_70.png)  
**transforms.Compose 归一化到 [-1.0, 1.0]**

```
transform2 = transforms.Compose([transforms.ToTensor()])
transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
```

### 二、torchvision 的 Transform

在深度学习时关于图像的数据读取：由于 Tensorflow 不支持与 numpy 的无缝切换，导致难以使用现成的 pandas 等格式化数据读取工具，造成了很多不必要的麻烦，而 pytorch 解决了这个问题。

pytorch 自定义读取数据和进行 Transform 的部分请见文档：  
[http://pytorch.org/tutorials/beginner/data_loading_tutorial.html](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

但是按照文档中所描述所完成的自定义 Dataset 只能够使用自定义的 Transform 步骤，而 torchvision 包中已经给我们提供了很多图像 transform 步骤的实现，为了使用这些已经实现的 Transform 步骤，我们可以使用如下方法定义 Dataset：

```
from __future__ import print_function, division 
import os 
import torch 
import pandas as pd 
from PIL import Image 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 

class FaceLandmarkDataset(Dataset): 
    def __len__(self) -> int: 
        return len(self.landmarks_frame)
		
    def __init__(self, csv_file: str, root_dir: str, transform=None) -> None: 
        super().__init__() 
        self.landmarks_frame = pd.read_csv(csv_file) 
        self.root_dir = root_dir 
        self.transform = transform 

    def __getitem__(self, index:int): 
        img_name = self.landmarks_frame.ix[index, 0] 
        img_path = os.path.join('./faces', img_name) 
        with Image.open(img_path) as img: 
            image = img.convert('RGB') 
        landmarks = self.landmarks_frame.as_matrix()[index, 1:].astype('float') 
        landmarks = np.reshape(landmarks,newshape=(-1,2)) 
        if self.transform is not None: 
            image = self.transform(image) 
        return image, landmarks 

########################以上为数据读取类（返回：image,landmarks）###############################
trans = transforms.Compose(transforms = [transforms.RandomSizedCrop(size=128), 
                                         transforms.ToTensor()]) 

face_dataset = FaceLandmarkDataset(csv_file='faces/face_landmarks.csv', 
				   root_dir='faces', transform= trans) 
loader = DataLoader(dataset = face_dataset, 
                    batch_size=4,
		    shuffle=True,
		    num_workers=4)
```

### 三、读取图像数据类

##### 3.1 class torchvision.datasets.ImageFolder 默认读取图像数据方法：

*   **`__init__`**（ 初始化）
    *   `classes, class_to_idx = find_classes(root)` : 得到分类的类别名（classes）和类别名与数字类别的映射关系字典（class_to_idx）  
        其中 classes (list): List of the class names.  
        其中 class_to_idx (dict): Dict with items (class_name, class_index).
    *   `imgs = make_dataset(root, class_to_idx)`得到 imags 列表。  
        其中 imgs (list): List of (image path, class_index) tuples  
        每个值是一个 tuple，每个 tuple 包含两个元素：图像路径和标签
*   **`__getitem__`**（图像获取）
    *   `path, target = self.imgs[index]` 获取图像（路径，标签）
    *   `img = self.loader(path)`数据读取。
    *   `img = self.transform(img)`数据、标签 转换成 tensor
    *   `target = self.target_transform(target)`
*   **`__len__`**（ 数据集数量）
    *   `return len(self.imgs)`

```
class ImageFolder(data.Dataset):
    """默认图像数据目录结构
    root
    .
    ├──dog
    |   ├──001.png
    |   ├──002.png
    |   └──...
    └──cat  
    |   ├──001.png
    |   ├──002.png
    |   └──...
    └──...
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        index (int): Index
	Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
```

图像获取 **`__getitem__`** 中，self.loader(path) 采用的是 default_loader，如下

```
def pil_loader(path):    # 一般采用pil_loader函数。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
```

##### 3.2 自定义数据读取方法

PyTorch 中和数据读取相关的类都要继承一个基类：torch.utils.data.Dataset。  
故需要改写其中的 **`__init__、__len__、__getitem__`** 等三个方法即可。

*   `__init__（）`初始化传入参数：
    *   img_path 里面为所有图像数据（包括训练和测试）  
        txt_path 里面有 train.txt 和 val.txt 两个文件：txt 文件中每行都是图像路径，tab 键，标签。
    *   其中 self.img_name 和 self.img_label 的读取方式就跟你数据的存放方式有关（需要调整的地方)
*   `__getitem__（）`依然采用 default_loader 方法来读取图像。
*   `Transform`中将每张图像都封装成 Tensor

```
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '',data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            """
	    关于json文件解析：
	    https://blog.csdn.net/wsp_1138886114/article/details/83302339
	    txt文件解析如下，具体文本解析具体分析，没有定数
            """
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label
#####################以上为图像数据读取，返回（img, label）#########################

# 保证image_datasets与torchvision.datasets.ImageFolder类返回的数据类型一样
image_datasets = {x: customData(img_path='/ImagePath',
                                txt_path=('/TxtFile/' + x + '.txt'),
                                data_transforms=data_transforms,
                                dataset=x) for x in ['train', 'val']}

#用torch.utils.data.DataLoader类，将这个batch的图像数据和标签都分别封装成Tensor。
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size=batch_size,
                                             shuffle=True) for x in ['train', 'val']}

# 模型保存
torch.save(model, 'output/resnet_epoch{}.pkl'.format(epoch))
```

[https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/#torchutilsdata](https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/#torchutilsdata)  
鸣谢  
[https://www.cnblogs.com/denny402/p/5096001.html](https://www.cnblogs.com/denny402/p/5096001.html)  
[https://blog.csdn.net/VictoriaW/article/details/72822005](https://blog.csdn.net/VictoriaW/article/details/72822005)  
[https://blog.csdn.net/hao5335156/article/details/80593349](https://blog.csdn.net/hao5335156/article/details/80593349)  
[https://blog.csdn.net/u014380165/article/details/78634829](https://blog.csdn.net/u014380165/article/details/78634829)