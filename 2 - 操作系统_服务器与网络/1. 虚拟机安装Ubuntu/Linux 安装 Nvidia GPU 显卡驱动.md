> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/inthat/article/details/106624245)

### 文章目录

*   [安装 Nvidia GPU 驱动](#Nvidia_GPU_1)
*   *   [什么是 nouveau 驱动？](#nouveau_6)
    *   [Centos7.7 安装 Nvidia GPU 驱动](#Centos77Nvidia_GPU_15)
    *   [Ubuntu 18.04 安装 Nvidia GPU 驱动](#Ubuntu_1804Nvidia_GPU_20)
    *   *   [准备工作](#_25)
        *   [开始安装](#_52)
        *   [安装 cuda](#cuda_113)
    *   [检测 NVIDIA 驱动是否成功安装](#NVIDIA_165)
    *   *   [集显与独显的切换](#_189)

安装 Nvidia GPU 驱动
================

[推荐]Linux 安装 NVIDIA 显卡驱动的正确姿势  
参考 URL：https://blog.csdn.net/wf19930209/article/details/81877822

什么是 nouveau 驱动？
---------------

nouveau，是一个自由及开放源代码显卡驱动程序，是为 Nvidia 的显示卡所编写，也可用于属于系统芯片的 NVIDIA Tegra 系列，此驱动程序是由一群独立的软件工程师所编写，Nvidia 的员工也提供了少许帮助。

该项目的目标为**利用逆向工程 Nvidia 的专有 Linux 驱动程序**来创造一个开放源代码的驱动程序。

所以 nouveau 开源驱动基本上是不能正常使用的，驱动性能较差。

> 总结：因此，我们一般需要安装官网原版驱动。

Centos7.7 安装 Nvidia GPU 驱动
--------------------------

Centos7.7 安装 Nvidia GPU 驱动及 CUDA 以及 tensorflow-GPU  
原文链接：https://blog.csdn.net/gy87900311/article/details/105074940

Ubuntu 18.04 安装 Nvidia GPU 驱动
-----------------------------

参考 URL: https://blog.csdn.net/wf19930209/article/details/81877822  
Ubuntu18.04 上安装 RTX 2080Ti 显卡驱动  
原文链接：https://blog.csdn.net/wangzi11111111/article/details/90447326

### 准备工作

1.  查看自己的机器的 GPU
    
    lspci | grep -i nvidia
    
    查看当前电脑的显卡型号  
    lshw -numeric -C display
    
2.  验证系统是否是受支持的 Linux 版本
    
    uname -m && cat /etc/issue
    
    到这里可以查看受支持的 Linux 版本：https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements
    
3.  官网 下载驱动  
    https://www.geforce.cn/drivers  
    到 NVIDIA 的官方驱动网站下载对应显卡的驱动程序，下载后的文件格式为 run。
    
4.  删除原有的 NVIDIA 驱动程序
    
    如果你没有安装过，或者已经卸载，可以忽略:
    
    sudo apt-get remove --purge nvidia*
    

### 开始安装

1、bios 禁用禁用 secure boot，也就是设置为 disable  
如果没有禁用 secure boot, 会导致 NVIDIA 驱动安装失败，或者不正常。

1、禁用 nouveau  
nouveau 是一个第三方开源的 Nvidia 驱动，一般 Linux 系统安装的时候都会默认安装这个驱动。这个驱动会与 nvidia 官方的驱动冲突，在安装 nvidia 驱动和 cuda 之前应该先禁用 nouveau  
查看系统是否正在使用 nouveau

```
lsmod | grep nouveau
```

如果有任何输出，那么就是 nouveau 在启用，需要关闭。，按照以下步骤：Ubuntu 中禁用方法：  
vi /etc/modprobe.d/blacklist.conf  
在最后一行添加：

```
blacklist nouveau
options nouveau modeset=0
```

这一条的含义是禁用 nouveau 第三方驱动，之后也不需要改回来。  
由于 nouveau 是构建在内核中的，所以要执行下面命令生效:

```
sudo update-initramfs -u
```

机器重启, **注意 需要重启**

```
sudo reboot now
```

重启之后，可以查看 nouveau 有没有运行:

```
lsmod | grep nouveau  # 没输出代表禁用生效
```

停止可视化桌面  
为了安装新的 Nvidia 驱动程序，我们需要停止当前的显示服务器。最简单的方法是使用 telinit 命令更改为运行级别 3。执行以下 linux 命令后，显示服务器将停止，因此请确保在继续之前保存所有当前工作（如果有）：

```
sudo telinit 3
```

2、安装驱动  
GPU 服务器需要正常工作需要安装正确的基础设施软件，对 NVIDIA 系列 GPU 而言，有两个层次的软件包需要安装：  
（1）驱动 GPU 工作的硬件驱动程序。  
（2）上层应用程序所需要的库

```
sudo chmod a+x NVIDIA-Linux-x86_64-440.82.run
sudo sh ./NVIDIA-Linux-x86_64-440.82.run --no-opengl-files
```

–no-opengl-files 参数必须加否则会循环登录，也就是 loop login  
参数介绍：  
–no-opengl-files 只安装驱动文件，不安装 OpenGL 文件。**这个参数最重要**  
–no-x-check 安装驱动时不检查 X 服务  
–no-nouveau-check 安装驱动时不检查 nouveau  
后面两个参数可不加。

cat /var/log/nvidia-install.log

如果没有问题，输入 nvidia-smi

nvidia-smi

### 安装 cuda

cuda 是 nvidia 公司推出的一套编程环境，包括驱动，sdk，toolkit 等。主要是用来进行计算加速，作为协处理器来进行使用。同时 cuda 有很多的库，如 cublas，cufft 等计算库，在用于科学计算和人工智能领域都有很好的加速效果。

主要应用除了日常视频编码解码，游戏等外，可以应用于计算加速方面。拿我所接触的行星模式模拟来讲，GPU 加速可以让我们模拟的物理计算过程获得很大的加速，加速科研产出。

```
一般使用，你可以跳过这一步！默认安装上面流程装完，私有就已经自动安装好了cuda。
```

1.  官网下载 cuda 本地可执行 run 文件  
    https://developer.nvidia.com/cuda-toolkit-archive
    
    注：根据 ubuntu 内核版本 gcc 版本以及 NVIDIA driver 版本进行选择，具体版本选择参考官方文档，附带补丁包也需下载及安装
    
2.  安装 cuda
    
    提升文件权限 sudo chmod a+x cuda…run -> sudo ./cuda…run -> 按 d 翻页 -> accept -> Install NVIDIA Driver? No 否则会覆盖之前安装的 Driver -> Install cuda toolkit? Yes -> toolkit localtion? default -> intall symbolic link? Yes -> Install samples? Yes -> samples location? default -> 安装完成
    
    直接运行文件即可（bash ./***.run），主意之前安装了驱动，所以在安装的时候选择不要安装驱动即可，其余的一路 y 下去。
    
    注意： 安装 CUDA 时一定使用 run 文件，**这样可以进行选择。不再选择安装驱动**，以及在弹出 xorg.conf 时选择 NO
    
3.  配置环境变量  
    vim ~/.bashrc
    
    ```
    在末尾添加如下内容（依据NVIDIA官方文档所述）
    
    export CUDA_HOME=/usr/local/cuda-10.0
    
    export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
    
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    
    之后source ~/.bashrc立即生效
    ```
    
    reboot 重启，并验证
    
4.  cuda 验证  
    首先，测试 cuda, nvcc 命令是否可用
    
    ```
    # cuda ; 按两下 tab 键
    cudafe                       cuda-gdb                     cuda-install-samples-9.0.sh
    cudafe++                     cuda-gdbserver               cuda-memcheck
    # nvcc --version
    ```
    
    接下来，用 cuda 例程测试，找到例程的安装目录，默认在 /root 下  
    只需要挑选其中的几个进行测试即可，比如
    
    ```
    # cd 1_Utilities/deviceQuery
    # make
    # ./deviceQuery
    ```
    
    至此，CUDA Toolkit 已经安装完成。
    

检测 NVIDIA 驱动是否成功安装
------------------

1.  使用 nvidia-setting 命令

```
apt install nvidia-settings
nvidia-setting
```

终端执行这个命令会调出图形化 NVIDIA 的驱动管理程序。  
如果出现这个界面可以看到 NVIDIA Driver Version：XXX.XX，这就代表 nvidia-setting 安装正常。

2.  使用 nvidia-smi 命令测试  
    英伟达系统管理接口（NVIDIA System Management Interface, 简称 nvidia-smi）是基于 NVIDIA Management Library (NVML) 的命令行管理组件, 旨在 (intened to ) 帮助管理和监控 NVIDIA GPU 设备。

```
nvidia-smi
```

执行这条命令将会打印出当前系统安装的 NVIDIA 驱动信息。

3.  命令搜索 集显和独显

```
lspci | grep VGA     # 查看集成显卡
lspci | grep NVIDIA  # 查看NVIDIA显卡
```

如果都能搜索到说明正常。

### 集显与独显的切换

1.  使用 nvidia-setting 图形化切换  
    终端执行 nvidia-setting, 在弹的界面中选择独显与集显:  
    ![](https://img-blog.csdnimg.cn/20200608193835406.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ludGhhdA==,size_16,color_FFFFFF,t_70)
2.  NVIDIA 提供了一个切换显卡的命令：

```
apt install nvidia-prime
sudo prime-select nvidia # 切换nvidia显卡
sudo prime-select intel  # 切换intel显卡
sudo prime-select query  # 查看当前使用的显卡
```

注意： 每一次切换显卡都需要重新启动电脑才能生效。