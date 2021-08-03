> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/zhengxiangwen/article/details/55148287)

一、使用 open 打开文件后一定要记得调用文件对象的 close() 方法。比如可以用 try/finally 语句来确保最后能关闭文件。

二、需要导入 import os

三、下面是逐行读取文件内容的三种方法：  

1、第一种方法：

```
f = open("foo.txt")               # 返回一个文件对象 
line = f.readline()               # 调用文件的 readline()方法 
while line: 
    print line,                   # 后面跟 ',' 将忽略换行符 
    #print(line, end = '')　      # 在 Python 3 中使用 
    line = f.readline() 
 
f.close()
```

2、第二种方法：

```
for line in open("foo.txt"): 
    print line,
```

3、第三种方法：

```
f = open("c:\\1.txt","r") 
lines = f.readlines()      #读取全部内容 ，并以列表方式返回
for line in lines 
    print line
```

四、一次性读取整个文件内容：

```
file_object = open('thefile.txt')
try:
     all_the_text = file_object.read()
finally:
     file_object.close()
```

五、区别对待读取文本 和 二进制：

1、如果是读取文本  

```
读文本文件
input = open('data', 'r')
#第二个参数默认为r
input = open('data')
```

2、如果是读取二进制

```
input = open('data', 'rb')
```

读固定字节

```
chunk = input.read(100)
```