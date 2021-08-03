# Python

enumerate()函数 - 枚举
--

>  原文地址 [www.runoob.com](https://www.runoob.com/python/python-func-enumerate.html)

用于将一个可遍历的数据对象 (如列表、元组或字符串) 组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。(下标，对象)

1. 语法

```
enumerate(sequence, [start=0])
```

2. 参数 

*   sequence -- 一个序列、迭代器或其他支持迭代对象。
*   start -- 下标起始位置。

3. 返回值

返回 enumerate(枚举) 对象。

4.  for 循环使用 enumerate

```
>>> seq = ['one', 'two', 'three'] 
>>> for i, element in enumerate(seq): 

... 	print i, element 
... 
0 one 
1 two 
2 three
```

## Python安装

### 安装路径

> 原文地址 [www.cnblogs.com](https://www.cnblogs.com/yrxns/p/11102985.html)

1、terminal : 

input: which Python 或者 which Python3

2、terminal:

input : python  --->import sys  ----> print sys.path 或者 input : python3  --->import sys  ----> print sys.path

### 版本号

python3 --version

# TensorFlow

## reshape - 重塑形状

tf.reshape & tf.shape 输出张量形状/重塑张量形状

https://www.tensorflow.org/api_docs/python/tf/reshape

Given `tensor`, this operation returns a new [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) that has the same values as `tensor` in the same order, except with a new shape given by `shape`.

```
#拉成一维列向量
t2 = tf.reshape(t1, [-1,])
```

tf.cast 类型转换

https://www.tensorflow.org/api_docs/python/tf/cast

# Numpy

## random.randint

在范围内生成随机整型数

> 原文地址 [blog.csdn.net](https://blog.csdn.net/u011851421/article/details/83544853)

```
numpy.random.randint(low, high=None, size=None, dtype='l')
```

函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即 [low, high)。  

如果没有写参数 high 的值，则返回 [0,low) 的值。

参数如下：

* low: int  

    生成的数值最低要大于等于 low。  hign = None 时，生成的数值要在 [0, low) 区间内）

*   high: int (可选)  
  
    如果使用这个值，则生成的数值在 [low, high) 区间。
    
*   size: int or tuple of ints(可选)  
  
    输出随机数的尺寸，比如 size = (m * n* k) 则输出同规模即 m * n* k 个随机数。默认是 None 的，仅仅返回满足要求的单一随机数。
    
*   dtype: dtype(可选)：  
  
    想要输出的格式。如`int64`、`int`等等。

输出：

*   out: int or ndarray of ints  
  
    返回一个随机数或随机数数组

**例子**

```
>>> np.random.randint(2, size=10)
array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
>>> np.random.randint(1, size=10)
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

```
>>> np.random.randint(5, size=(2, 4))
array([[4, 0, 2, 1],
       [3, 2, 2, 0]])
```

```
>>>np.random.randint(2, high=10, size=(2,3))
array([[6, 8, 7],
       [2, 5, 2]])
```

# PIL库

图像库PIL的类Image

## Image.fromarray与asarray

>  原文地址 [blog.csdn.net](https://blog.csdn.net/ybcrazy/article/details/81206411)

1. PIL image 转换成 array

```
img = np.asarray(image)
```

需要注意的是，如果出现 read-only 错误，并不是转换的错误，一般是你读取的图片的时候，默认选择的是 "r","rb" 模式有关。

修正的办法:　手动修改图片的读取状态

```
img.flags.writeable = True  # 将数组改为读写模式
```

2. array 转换成 image

`Image.fromarray(np.uint8(img))`

