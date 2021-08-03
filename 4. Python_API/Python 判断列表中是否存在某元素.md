> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [www.cnblogs.com](https://www.cnblogs.com/karkash/p/13541073.html)

**成员运算符**
---------

<table><thead><tr><th>运算符</th><th>描述</th></tr></thead><tbody><tr><td>in</td><td>如果在指定的序列中找到值返回 True，否则返回 False</td></tr><tr><td>not in</td><td>如果在指定的序列中没有找到值返回 True，否则返回 False</td></tr></tbody></table>

### **实例：**

[![](http://common.cnblogs.com/images/copycode.gif)](javascript:void(0); "复制代码")

[![](http://common.cnblogs.com/images/copycode.gif)](javascript:void(0); "复制代码")

**结果：**

1   在列表 lista 中  
cf   在列表 lista 中

is 与 == 区别：  
is 用于判断两个变量引用对象是否为同一个， == 用于判断引用变量的值是否相等

[![](http://common.cnblogs.com/images/copycode.gif)](javascript:void(0); "复制代码")

```
#-*- coding:utf-8 -*- python 3.6.2
a=1
b=1
lista=[1,'5','s','cf']
listb=[1,'5','s','cf']

if a is b:
    print('a=b')
if listb is lista:
    print('lista is listb')
if lista == listb:
    print('lista=listb')

```

[![](http://common.cnblogs.com/images/copycode.gif)](javascript:void(0); "复制代码")

**结果：**  

a=b  
lista=listb