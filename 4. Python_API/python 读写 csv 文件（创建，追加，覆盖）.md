> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/lwgkzl/article/details/82147474)

总述：
---

这篇博客讲述 python 怎样创建，读写，追加 csv 文件

创建：
---

利用 csv 包中的 writer 函数，如果文件不存在，会自动创建，需要注意的是，文件后缀一定要是. csv，这样才会创建 csv 文件

这里创建好文件，将 csv 文件的头信息写进了文件。

```
import csv
def create_csv():
    path = "aa.csv"
    with open(path,'wb') as f:
        csv_write = csv.writer(f)
        csv_head = ["good","bad"]
        csv_write.writerow(csv_head)
```

追加：
---

在 python 中，以 a + 的方式打开，是追加

```
def write_csv():
    path  = "aa.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = ["1","2"]
        csv_write.writerow(data_row)
```

读：
--

利用 csv.reader 可以读 csv 文件，然后返回一个可迭代的对象 csv_read，我们可以直接从 csv_read 中取数据

```
def read_csv():
    path = "aa.csv"
    with open(path,"rb") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            print line
```

附加：
---

python 利用 open 打开文件的方式：

**w**：以写方式打开，   
**a**：以追加模式打开 (从 EOF 开始, 必要时创建新文件)   
**r+**：以读写模式打开   
**w+**：以读写模式打开 (参见 w)   
**a+**：以读写模式打开 (参见 a)   
**rb**：以二进制读模式打开   
**wb**：以二进制写模式打开 (参见 w)   
**ab**：以二进制追加模式打开 (参见 a)   
**rb+**：以二进制读写模式打开 (参见 r+)   
**wb+**：以二进制读写模式打开 (参见 w+)   
**ab+**：以二进制读写模式打开 (参见 a+)