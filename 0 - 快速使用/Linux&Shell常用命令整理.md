## 远程连接服务器 Server

```shell
# 威富
ssh wyfubt@193.168.1.213
ssh wyfubt@server.natappfree.cc -p 37420
# 半导体
ssh wangyangfan@172.16.0.92
```

## Linux包管理器

建议大家尽快适应并开始首先使用 apt。不仅因为广大 Linux 发行商都在推荐 apt，更主要的还是它提供了 Linux 包管理的必要选项。

###  Conda

Conda管理虚拟环境

```shell
# list所有虚拟环境
conda env list
# 创建虚拟环境
conda create -n tf tensorflow
# 激活虚拟环境
conda activate tf
# 关闭虚拟环境
conda deactivate  
# conda升级所有包
conda upgrade --all -y
# 在某环境下安装包
conda install -n DeepsortYolov5 pandas
# list某环境下安装的包
conda list -n env_name
# 在某环境下安装reqirements
conda install -n DeepsortYolov5 --yes --file requirements.txt
```

### pip

```
#更新pip
pip install --upgrade pip
```

pip 批量导出包含环境中所有组件的 requirements.txt 文件

```
pip freeze > requirements.txt
```

pip 批量安装 requirements.txt 文件中包含的组件依赖

```
pip install -r requirements.txt
```

conda 批量导出包含环境中所有组件的 requirements.txt 文件

```
conda list -e > requirements.txt
```

Conda 批量安装 requirements.txt 文件中包含的组件依赖

```
conda install --yes --file requirements.txt
```

## Debug

> To run a command as administrator (user "root"), use "sudo <command>".

原因：权限不够，在你输入的命令前面加上sudo,如：sudo xxxx

1. 保持现状：如果你要运行 "ls" 这条命令,那么你就在命令行上输入 "sudo ls",然后会提示你输入密码,那么你就输入密码。
2. 但如果你想要保持在超级用户(也就是root,相当于管理员)状态的话,可以执行"sudo -s"然后输入密码,你就能作为超级用户执行命令了而不必在每一个命令前加"sudo"了.

## VIM

1. 进入vim模式

```
Vim /est/ssh/sshd_config
```

2. vim编辑模式

```
# 进入编辑模式
>> i

# 退出编辑模式
>> esc
```

3. 退出vim模式

```sh
# 退出vim模式
>> :q
# 强制退出
>> :q!
# 保存修改
>> :w
# 强制保存
>> :w!
#保存并退出
>> :wq
```

## 查询ip地址

查询ip地址：用“ifconfig”命令 - 观察inet

## 文件与路径

命令pwd用于显示当前所在目录

删除文件

```
rm –rf命令
```

相对路径：基于当前路径

```
cd test
# ..表示上一级菜单
```

绝对路径：斜杠开头

```
cd /etc/ssh
```

##  文件树状结构 - Tree

>  原文地址 [www.runoob.com](https://www.runoob.com/linux/linux-comm-tree.html)

Linux tree 命令用于以树状图列出目录的内容。

执行 tree 指令，它会列出指定目录下的所有文件，包括子目录里的文件。

以树状图列出当前目录结构。可直接使用如下命令：

```
# install
sudo apt-get install tree 
# Usage
tree -N
tree --help
```



## 重命名

rename OldName NewName

>  原文地址 [blog.csdn.net](https://blog.csdn.net/xudadada/article/details/109085291)

坚持使用 Linux 终端，该如何从终端下载文件？

Linux 中没有下载命令，但是有几个用于下载文件的 Linux 命令。

 wget 命令 - 用终端下载文件
============================

wget 是非交互式的，可以轻松在后台运行。这意味着您可以轻松地在脚本中使用它，甚至可以构建 uGet 下载管理器之类的工具。

大多数 Linux 发行版都预装有 wget。大多数发行版的存储库中也提供了该软件，您可以使用发行版的程序包管理器轻松安装它。

对于 Linux 和类UNIX 的系统，wget 可能是最常用的命令行下载管理器。您可以使用 wget 下载单个文件，多个文件，整个目录，甚至整个网站。

0. 安装 wget

1. **下载&命名指定目录**

把 index.html 文件保存到 "/root/test" 目录下.

```
wget -P /root/test "http://www.baidu.com/index.html"  
```

2. **下载&指定命名**

把 index.hmtl 保存到当前目录, 命令为 "baidu.html"

```
wget -O "baidu.html" "http://www.baidu.com/index.html"  
```

3. **下载多个文件**

要下载多个文件，可以将它们的URL保存在一个文本文件中，并提供该文本文件作为wget的输入，如下所示：

```
wget -i download_files
```

4. **恢复不完整的下载**

如果由于某些原因按下 C 放弃了下载，则可以使用选项 - c 恢复上一次下载。

```
wget -c
```



## Linux shell批处理命令

1. 新建脚本 touch natstart.sh

2. 写入命令 vim natstart.sh

   ```sh
    cd app
   ./Natapp -authtoken=de714e7ebdbbf2d5
   ```

3. 执行命令 sh natstart.sh



## 定位进程与杀进程

 原文地址 [www.linuxprobe.com](https://www.linuxprobe.com/linux-kill-manager.html)

**定位进程：top 和 ps 命令**

top命令：能够知道到所有当前正在运行的进程有哪些。

```
top
```

Chrome 浏览器反映迟钝，依据top 命令显示，我们能够辨别的有四个 Chrome 浏览器的进程在运行，进程的 pid 号分别是 3827、3919、10764 和 11679。这个信息是重要的，可以用一个特殊的方法来结束进程。

尽管 top 命令很是方便，但也不是得到你所要信息最有效的方法。 

你明确要杀死的 Chrome 进程，并且你也不想看 top 命令所显示的实时信息。 鉴于此，使用 ps 命令然后用 grep 命令来过滤出输出结果。

```
ps aux | grep chrome
```

**结束进程**

现在我们开始结束进程的任务。我们有两种可以帮我们杀死错误的进程的信息。

你用哪一个将会决定终端命令如何使用，通常有两个命令来结束进程：

*   kill - 通过进程 ID 来结束进程
*   killall - 通过进程名字来结束进程

最经常使用的结束进程的信号是：

<table><thead><tr><th>Signal Name</th><th>Single Value</th><th>Effect</th></tr></thead><tbody><tr><td>SIGHUP</td><td>1</td><td>挂起</td></tr><tr><td>SIGINT</td><td>2</td><td>键盘的中断信号</td></tr><tr><td>SIGKILL</td><td>9</td><td>发出杀死信号</td></tr><tr><td>SIGTERM</td><td>15</td><td>发出终止信号</td></tr><tr><td>SIGSTOP</td><td>17, 19, 23</td><td>停止进程</td></tr></tbody></table>

好的是，你能用信号值来代替信号名字。所以你没有必要来记住所有各种各样的信号名字。

所以，让我们现在用 kill 命令来杀死 Chrome 浏览器的进程。这个命令的结构是：

```
kill SIGNAL PID
```

这里 SIGNAL 是要发送的信号，PID 是被杀死的进程的 ID。我们已经知道，来自我们的 ps 命令显示我们想要结束的进程 ID 号是 3827、3919、10764 和 11679。所以要发送结束进程信号，我们输入以下命令：

```
kill -9 3827
kill -9 3919
kill -9 10764
kill -9 11679
```

一旦我们输入了以上命令，Chrome 浏览器的所有进程将会成功被杀死。

我们有更简单的方法！如果我们已经知道我们想要杀死的那个进程的名字，我们能够利用 killall 命令发送同样的信号

```
killall -9 chrome
```

附带说明的是，上边这个命令可能不能捕捉到所有正在运行的 Chrome 进程。如果，运行了上边这个命令之后，你输入 ps aux | grep chrome 命令过滤一下，看到剩下正在运行的 Chrome 进程有那些，最好的办法还是回到 kIll 命令通过进程 ID 来发送信号值 9 来结束这个进程。

**举例**

正如你看到的，杀死错误的进程并没有你原本想的那样有挑战性。当我让一个顽固的进程结束的时候，我趋向于用 killall 命令来作为有效的方法来终止，然而，当我让一个真正的活跃的进程结束的时候，kill 命令是一个好的方法。

```
 ps -ef|grep natapp
```

![image-20210504000914925](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/05_04_image-20210504000914925.png)

杀进程时，填写框选出来的内容，最下一行代表查找程序本身, 忽略掉。

```
kill -9 pid
```

![image-20210504003054136](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/05_04_image-20210504003054136.png)



## &&（命令执行控制）    

语法格式如下：        

　　command1 && command2 [&& command3 ...]    

1 命令之间使用 && 连接，实现逻辑与的功能。  

2 只有在 && 左边的命令返回真（命令返回值 $? == 0），&& 右边的命令才会被执行。  

3 只要有一个命令返回假（命令返回值 $? == 1），后面的命令就不会被执行。  



## pathlib

### as_posix()

将 Windows 路径分隔符 ‘\’ 改为 Unix 样式 ‘/’

### [python shutil.copy()用法](https://www.cnblogs.com/liuqi-beijing/p/6228561.html)

shutil.copyfile(src, dst)：复制文件内容（不包含元数据）从src到dst。

DST必须是完整的目标文件名;



## Debug

Q1:开启内网穿透却无法进行ssh连接

https://blog.csdn.net/qq_36441027/article/details/81708726

![image-20210512204320098](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/05_12_image-20210512204320098.png)

本机连接服务器，会保存服务器的公钥，包括github或其他服务器。但是ip端口号对应的公钥被更改，就可能被误认为被攻击，无法连接，需要删除曾经在~/.ssh/known_hosts对应的公钥

![image-20210512204652948](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/05_12_image-20210512204652948.png)

Q2:无法连接SFTP

设置为密码登陆



快捷键

显示桌面：Ctrl + Win + d

切换输入法：win+space

Root登录： 

https://www.youtube.com/watch?v=qziz2iEyLcc

安装网络app

chmod a+x <packagename> //权限

./<packagename> //运行

卸载

dpkg --list | grep <package_name>

返回

apt remove <package_name>

无法定位软件包？需要更新源

https://www.jianshu.com/p/7916c6787b4f

ubuntu ： 无法安全地用该源进行更新，所以默认禁用该源

解决占用端口问题：sudo lsof -i :7890



vscode在root下打不开，换成普通用户

安装搜狗输入法：https://pinyin.sogou.com/linux/help.php


