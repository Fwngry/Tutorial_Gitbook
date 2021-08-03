> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/283805135)

```
# 导入相关模块和包
import cv2 as cv
import numpy as np


# 定义抠图函数
def get_face(image):
    face = image[40:600, 110:400]
    cv.imshow('get_face', face)


# 读取图片
src = cv.imread(r'C:\Users\Administrator.PC-20201106KUIO\Desktop\lena.jpg')
# 展示图片
cv.imshow('artwork mester', src)
# 抠图函数
get_face(src)
# 等待用户按下任意键
cv.waitKey(0)
# 释放内存
cv.destroyAllWindows()
```

![](https://pic2.zhimg.com/v2-21caa5b333081dc06a139350650f23c9_r.jpg)