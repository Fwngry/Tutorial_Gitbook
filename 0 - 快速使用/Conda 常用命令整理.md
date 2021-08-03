> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/menc15/article/details/71477949)

主要参考 Anaconda 官方指南 Using Conda：[https://conda.io/docs/using/index.html](https://conda.io/docs/using/index.html)

环境：Win10 64bit with conda 4.3.14  
以下命令均在 windows 命令行中输入。一般来讲，无论是在 Linux，OS X 还是在 windows 系统中，在命令行窗口中输入的 conda 命令基本是一致的，除非有特别标注。

0. 获取版本号
--------

```
conda --version
```

或

```
conda -V
```

1. 获取帮助
-------

```
conda --help
conda -h
```

查看某一命令的帮助，如 update 命令及 remove 命令

```
conda update --help
conda remove --help
```

同理，以上命令中的`--help`也可以换成`-h`。

2. 环境管理
-------

查看环境管理的全部命令帮助

```
conda env -h
```

![](https://img-blog.csdn.net/20170509115051844?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbWVuYzE1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

创建环境

```
conda create --name your_env_name
```

输入`y`确认创建。

创建制定 python 版本的环境

```
conda create --name your_env_name python=2.7
conda create --name your_env_name python=3
conda create --name your_env_name python=3.5
```

创建包含某些包的环境

```
conda create --name your_env_name numpy scipy
```

创建指定 python 版本下包含某些包的环境

```
conda create --name your_env_name python=3.5 numpy scipy
```

列举当前所有环境

```
conda info --envs
conda env list
```

进入某个环境

```
activate your_env_name
```

退出当前环境

```
deactivate
```

复制某个环境

```
conda create --name new_env_name --clone old_env_name
```

删除某个环境

```
conda remove --name your_env_name --all
```

3. 分享环境
-------

如果你想把你当前的环境配置与别人分享，这样 ta 可以快速建立一个与你一模一样的环境（同一个版本的 python 及各种包）来共同开发 / 进行新的实验。一个分享环境的快速方法就是给 ta 一个你的环境的`.yml`文件。

首先通过`activate target_env`要分享的环境`target_env`，然后输入下面的命令会在当前工作目录下生成一个`environment.yml`文件，

```
conda env export > environment.yml
```

小伙伴拿到`environment.yml`文件后，将该文件放在工作目录下，可以通过以下命令从该文件创建环境

```
conda env create -f environment.yml
```

`.yml`是这个样子的  
![](https://img-blog.csdn.net/20170509143617291?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbWVuYzE1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

当然，你也可以手写一个`.yml`文件用来描述或记录你的 python 环境。

4. 包管理
------

列举当前活跃环境下的所有包

```
conda list
```

列举一个非当前活跃环境下的所有包

```
conda list -n your_env_name
```

为指定环境安装某个包

```
conda install -n env_name package_name
```

如果不能通过 conda install 来安装，文档中提到可以从 Anaconda.org 安装，但我觉得会更习惯用 pip 直接安装。pip 在 Anaconda 中已安装好，不需要单独为每个环境安装 pip。如需要用 pip 管理包，activate 环境后直接使用即可。

TBA