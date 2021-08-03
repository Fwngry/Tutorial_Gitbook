> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/eddy23513/article/details/81366507)

1. 安装 SSH

```
如果你用的是redhat，fedora，centos等系列linux发行版，那么敲入以下命令：
sudo yum install sshd 或
sudo yum install openssh-server（由osc网友 火耳提供）

如果你使用的是debian，ubuntu，linux mint等系列的linux发行版，那么敲入以下命令：
sudo apt-get install sshd 或
sudo apt-get install openssh-server（由osc网友 火耳提供）

然后按照提示，安装就好了。
```

![](https://img-blog.csdn.net/20180802195113900?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2VkZHkyMzUxMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

2. 开启 ssh 服务

```
service sshd start
```

![](https://img-blog.csdn.net/20180802195201361?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2VkZHkyMzUxMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

3. 卸载服务

```
如果你用的是redhat，fedora，centos等系列linux发行版，那么敲入以下命令：
yum remove sshd
如果你使用的是debian，ubuntu，linux mint等系列的linux发行版，那么敲入以下命令：
sudo apt-get –purge remove sshd
```

![](https://img-blog.csdn.net/20180802195257771?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2VkZHkyMzUxMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)