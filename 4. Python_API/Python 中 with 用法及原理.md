> 原文地址 [blog.csdn.net](https://blog.csdn.net/u012609509/article/details/72911564)

前言
--

> 代替try···catch···

with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的 “清理” 操作，释放资源，比如文件使用后自动关闭／线程中锁的自动获取和释放等。

问题引出
----

如下代码：

```
file = open("１.txt")
data = file.read()
file.close()
```

上面代码存在２个问题： 
（１）文件读取发生异常，但没有进行任何处理； 
（２）可能忘记关闭文件句柄；

改进：try···catch···
--

```
try:
    f = open('xxx')
except:
    print('fail to open')
    exit(-1)
try:
    do something
except:
    do something
finally:
    f.close()
```

## 改进：with···as

而使用 with 的话，能够减少冗长，还能自动处理上下文环境产生的异常。如下面代码：

```
with open("１.txt") as file:
    data = file.read()
```

with 工作原理
---------

（１）紧跟 with 后面的语句被求值后，返回对象的 “–enter–()” 方法被调用，这个方法的返回值将被赋值给 as 后面的变量； 
（２）当 with 后面的代码块全部被执行完之后，将调用前面返回对象的 “–exit–()” 方法。 
with 工作原理代码示例：

```
class Sample:
    def __enter__(self):
        print "in __enter__"
        return "Foo"
    def __exit__(self, exc_type, exc_val, exc_tb):
        print "in __exit__"
def get_sample():
    return Sample()
with get_sample() as sample:
    print "Sample: ", sample
```

代码的运行结果如下：

```
in __enter__
Sample:  Foo
in __exit__
```

可以看到，整个运行过程如下： 
（１）**enter**() 方法被执行； 
（２）**enter**() 方法的返回值，在这个例子中是”Foo”，赋值给变量 sample； 
（３）执行代码块，打印 sample 变量的值为”Foo”； 
（４）**exit**() 方法被调用；

总结
--

实际上，在 with 后面的代码块抛出异常时，**exit**() 方法被执行。开发库时，清理资源，关闭文件等操作，都可以放在 **exit**() 方法中。 
总之，with-as 表达式极大的简化了每次写 finally 的工作，这对代码的优雅性是有极大帮助的。 
如果有多项，可以这样写：

```
With open('1.txt') as f1, open('2.txt') as f2:
    do something
```

参考网址
----

[http://blog.kissdata.com/2014/05/23/python-with.html](http://blog.kissdata.com/2014/05/23/python-with.html)