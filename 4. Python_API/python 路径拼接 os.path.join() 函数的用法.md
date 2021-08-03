> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [www.cnblogs.com](https://www.cnblogs.com/an-ning0920/p/10037790.html)

os.path.join() 函数：连接两个或更多的路径名组件

                         1. 如果各组件名首字母不包含’/’，则函数会自动加上

　　　　　　　　　2. 如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃

　　　　　　　　　3. 如果最后一个组件为空，则生成的路径以一个’/’分隔符结尾

### Demo1

import os

Path1 = 'home'  
Path2 = 'develop'  
Path3 = 'code'

Path10 = Path1 + Path2 + Path3  
Path20 = os.path.join(Path1,Path2,Path3)  
print ('Path10 =',Path10)  
print ('Path20 =',Path20)

**输出**

Path10 = homedevelopcode  
Path20 = home\develop\code

**--------------------------------------------------------------**

### Demo2

import os

Path1 = '/home'  
Path2 = 'develop'  
Path3 = 'code'

Path10 = Path1 + Path2 + Path3  
Path20 = os.path.join(Path1,Path2,Path3)  
print ('Path10 =',Path10)  
print ('Path20 =',Path20)   
**输出**

Path10 = /homedevelopcode  
Path20 = /home\develop\code

**--------------------------------------------------------------**

### Demo3

import os

Path1 = 'home'  
Path2 = '/develop'  
Path3 = 'code'

Path10 = Path1 + Path2 + Path3  
Path20 = os.path.join(Path1,Path2,Path3)  
print ('Path10 =',Path10)  
print ('Path20 =',Path20) 

**输出**

Path10 = home/developcode  
Path20 = /develop\code

**--------------------------------------------------------------**

### Demo4

import os

Path1 = 'home'  
Path2 = 'develop'  
Path3 = '/code'

Path10 = Path1 + Path2 + Path3  
Path20 = os.path.join(Path1,Path2,Path3)  
print ('Path10 =',Path10)  
print ('Path20 =',Path20)

**输出**

Path10 = homedevelop/code  
Path20 = /code

转：https://blog.csdn.net/hduxiejun/article/details/80289476