> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.51cto.com](https://blog.51cto.com/zaa47/2596875)

> ESXi GPU 直通，​最近在协助客户进行 ESXiGPU 直通配置，因为没有相关文档指导，跳了不少坑，今天把配置过程整理下，留作纪念，同时也希望可以帮助你尽快从坑里爬出来。

© 著作权归作者所有：来自 51CTO 博客作者 zaa47 的原创作品，如需转载，请注明出处，否则将追究法律责任  
https://blog.51cto.com/zaa47/2596875

​ 最近在协助客户进行 ESXi GPU 直通配置，因为没有相关文档指导，跳了不少坑，今天把配置过程整理下，留作纪念。

物理机及虚拟机配置
---------

​ 参考 vmware 及 NVIDIA 官网介绍，进行 ESXi GPU 直通配置时，为了避免一系列的报错，请按照以下要求完成配置。

> 简单点来说就是：
> 
> 物理机与虚拟机建议全部使用 EFI 引导。
> 
> ESXi 软件建议使用 6.7 及以上版本，操作系统安装 64-bit 的。

### 物理机配置

*   物理机使用`EFI`引导模式；
*   若 GPU 需要 16 GB 或更多的内存映射（BAR1 Memory），需要在物理机 bios 中启用 GPU 直通，设置项名称通常为 Above 4G decoding、Memory mapped I/O above 4GB 或 PCI 64-bit resource handing above 4G；
    
*   BIOS 中启用虚拟化功能： Intel Virtualization Technology for Directed I/O (VT-d) 或 AMD I/O Virtualization Technology (IOMMU)；

### ESXi 虚拟机设置

*   建议虚拟机系统为 64-bit 操作系统；
    
*   If the total BAR1 memory exceeds 256 Mbytes, EFI boot must be enabled for the VM.
    
    **Note:** To determine the total BAR1 memory, run `nvidia-smi -q` on the host.
    
*   To enable 64-bit Memory Mapped I/O (MMIO) add this line to the virtual machine vmx file:
    
    `pciPassthru.use64bitMMIO="TRUE"`
    
*   Memory Mapped I/O (MMIO) 大小调整：建议调整为（n*GPU 显存）向上舍入到下一个 2 次幂。
    
    示例：`pciPassthru.64bitMMIOSizeGB ="64"`
    
    *   两个 16G 显存 GPU，2 x 16 GB = 32，将 32 GB 向上舍入到下一个 2 次幂，所需的内存量为 64 GB。
        
    *   三个 16G 显存 GPU，3 x 16 GB = 48，将 48 GB 向上舍入到下一个 2 次幂，所需的内存量为 64 GB。
    
    _或者直接设置为虚拟机分配的所有 GPU 显存大小的两倍，2*n*GPU 显存（单位为 GB）_
    
*   虚拟机内存最小值建议为分配的所有 GPU 显存总大小的 1.5 倍。

> ESXI 6.5 以下版本注意事项:
> 
> *   Set the ESXi hosts BIOS to allow PCI mapping above 4GB and below 16TB(比如物理机 bios 中 MMIO High Base 设置为 4T).
> *   In UEFI BIOS mode, a virtual machines's total BAR allocation is limited to 32GB.

ESXi GPU 直通兼容性列表
----------------

​ ESXi 与 GPU 直通的兼容性列表一定提前查询下，比如 NVIDIA Tesla V100S 与 ESXi 6.0 是不兼容的，这个也是我本次安装踩坑之一，这个坑希望你不要掉进去。

​ 兼容性列表可以在 [GPU 直通兼容性查询网站](https://www.vmware.com/resources/compatibility/search.php?deviceCategory=vsga)，选择`Shared Pass-Through Graphics`项目后进行查询，该网站打开后经常没反应，不要一直纠结这个，看我下面附的图片吧。

### NVIDIA GPU 兼容性列表

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/20210118142027.png)

### AMD GPU 兼容性列表

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/20210118142142.png)

实际配置演示
------

### 软硬件环境介绍

​ 这里介绍下我本次配置所用的软硬件环境：

*   GPU：NVIDIA Tesla V100 与 Tesla V100S 各一块，均为 32G 显存 GPU。
*   虚拟化软件：ESXi 6.7 U3
*   虚拟机操作系统：CentOS 7.5-64-bit

### 物理机配置

首先开机进入 bios，提前修改物理机 bios 设置：

*   Above 4G decoding - Enable
*   Intel Virtualization Technology for Directed I/O (VT-d) - Enable
*   MMIO High Base - 默认 56T（若为 ESXi 6.5 以下版本注意修改为 4G-16T 之间的值，如 4T）

### ESXi 6.7 安装

​ 可以使用服务器自带的虚拟光驱或刻录 U 盘进行 ESXi 安装，本次以 U 盘安装进行示例，建议使用 Rufu 工具进行 U 盘刻录。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/U%E7%9B%98%E5%88%BB%E5%BD%95-20210118180427.png)

​ 服务器开机从 U 盘启动并完成 ESXi 安装，如图为本次完成安装的 ESXi 软件版本。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/ESXI20210118160609.png)

### GPU 切换直通模式

​ 安装完 ESXi 软件后，首先需要将 GPU 切换为直通模式，切换方法为：导航界面选择`管理`--->`硬件`--->`PCI设备`，搜索框输入`nvidia`筛选出 GPU 设备，勾选后，点击`切换直通`。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/20210118155302.png)

GPU 切换直通后，需要`重新引导主机`使配置生效：

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/20210118155417.png)

重新引导主机后，GPU 直通变为`活动`状态，表示 GPU 切换直通成功。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/GPU%E7%9B%B4%E9%80%9A-20210118161111.png)

### 虚拟机创建及配置

**创建虚拟机**

​ 导航栏选择`虚拟机`--->`创建/注册虚拟机`并修改虚拟机配置。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E5%88%9B%E5%BB%BA%E8%99%9A%E6%8B%9F%E6%9C%BA1-20210118172456.png)

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E5%88%9B%E5%BB%BA%E8%99%9A%E6%8B%9F%E6%9C%BA2-20210118172527.png)

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E9%80%89%E6%8B%A9%E5%AD%98%E5%82%A8-20210118172625.png)

**选择系统安装介质**

​ 本次通过上传 iso 镜像的方式进行虚拟机系统安装，`虚拟硬件`--->`CD/DVD驱动器1`中选择`数据存储ISO文件`，上载 ISO 镜像并选择。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E9%80%89%E6%8B%A9%E7%B3%BB%E7%BB%9F%E9%95%9C%E5%83%8F-20210118171858.png)

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAACbklEQVRoQ+2aMU4dMRCGZw6RC1CSSyQdLZJtKQ2REgoiRIpQkCYClCYpkgIESQFIpIlkW+IIcIC0gUNwiEFGz+hlmbG9b1nesvGW++zxfP7H4/H6IYzkwZFwQAUZmpJVkSeniFJKA8ASIi7MyfkrRPxjrT1JjZ8MLaXUDiJuzwngn2GJaNd7vyP5IoIYY94Q0fEQIKIPRGS8947zSQTRWh8CwLuBgZx479+2BTkHgBdDAgGAC+fcywoyIFWqInWN9BSONbTmFVp/AeA5o+rjKRJ2XwBYRsRXM4ZXgAg2LAPzOCDTJYQx5pSIVlrC3EI45y611osMTHuQUPUiYpiVooerg7TWRwDAlhSM0TuI+BsD0x4kGCuFSRVzSqkfiLiWmY17EALMbCAlMCmI6IwxZo+INgQYEYKBuW5da00PKikjhNNiiPGm01rrbwDwofGehQjjNcv1SZgddALhlJEgwgJFxDNr7acmjFLqCyJuTd6LEGFttpmkYC91Hrk3s1GZFERMmUT01Xv/sQljjPlMRMsxO6WULwnb2D8FEs4j680wScjO5f3vzrlNJszESWq2LYXJgTzjZm56MCHf3zVBxH1r7ftU1splxxKYHEgoUUpTo+grEf303rPH5hxENJqDKQEJtko2q9zGeeycWy3JhpKhWT8+NM/sufIhBwKI+Mta+7pkfxKMtd8Qtdbcx4dUQZcFCQ2I6DcAnLUpf6YMPxhIDDOuxC4C6djoQUE6+tKpewWZ1wlRkq0qUhXptKTlzv93aI3jWmE0Fz2TeujpX73F9TaKy9CeMk8vZusfBnqZ1g5GqyIdJq+XrqNR5AahKr9CCcxGSwAAAABJRU5ErkJggg==)

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E4%B8%8A%E8%BD%BDiso%E9%95%9C%E5%83%8F-20210118172345.png)

**添加直通 GPU 并预留所有内存**

​ `添加其他设置`--->`PCI设备`

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E6%B7%BB%E5%8A%A0%E7%9B%B4%E9%80%9AGPU-20210118164926.png)

​ 如图，添加两块 GPU，分别为 Tesla V100 和 Tesla V100S，并在`新PCI设备`选项下点击`预留所有内存`。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E6%B7%BB%E5%8A%A0%E7%9B%B4%E9%80%9AGPU-20210118165407.png)

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E6%B7%BB%E5%8A%A0%E7%9B%B4%E9%80%9AGPU-20210118165744.png)

**修改虚拟机内存**

​ `虚拟硬件`--->`内存`，建议设置最小内存为虚拟机所分配 GPU 显存总大小的 1.5 倍。确保已勾选**预留所有客户机内存 (全部锁定)**

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/esxi%E5%86%85%E5%AD%98%E8%AE%BE%E7%BD%AE-20210118164535.png)

**修改 MMIO 相关参数**

`虚拟机选项`--->`高级`--->`编辑配置`，添加以下参数：

本次添加两块显存为 32G 的 GPU，所以设置`pciPassthru.64bitMMIOSizeGB`的值为`2*32`并向上舍入到下一个 2 的次幂，即`128`.

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E8%99%9A%E6%8B%9F%E6%9C%BA%E9%80%89%E9%A1%B9-20210118170552.png)

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/20210119151612.png)

**修改虚拟机引导选项**

编辑虚拟机，修改`虚拟机选项`--->`引导选项`为`EFI`。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E8%99%9A%E6%8B%9F%E6%9C%BA%E5%BC%95%E5%AF%BC%E9%80%89%E9%A1%B9-20210118161714.png)

**开始安装虚拟机**

​ 自定义设置中完成 CPU、内存、GPU、引导选项等各种设置后，点击完成开始安装虚拟机即可。

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAACbklEQVRoQ+2aMU4dMRCGZw6RC1CSSyQdLZJtKQ2REgoiRIpQkCYClCYpkgIESQFIpIlkW+IIcIC0gUNwiEFGz+hlmbG9b1nesvGW++zxfP7H4/H6IYzkwZFwQAUZmpJVkSeniFJKA8ASIi7MyfkrRPxjrT1JjZ8MLaXUDiJuzwngn2GJaNd7vyP5IoIYY94Q0fEQIKIPRGS8947zSQTRWh8CwLuBgZx479+2BTkHgBdDAgGAC+fcywoyIFWqInWN9BSONbTmFVp/AeA5o+rjKRJ2XwBYRsRXM4ZXgAg2LAPzOCDTJYQx5pSIVlrC3EI45y611osMTHuQUPUiYpiVooerg7TWRwDAlhSM0TuI+BsD0x4kGCuFSRVzSqkfiLiWmY17EALMbCAlMCmI6IwxZo+INgQYEYKBuW5da00PKikjhNNiiPGm01rrbwDwofGehQjjNcv1SZgddALhlJEgwgJFxDNr7acmjFLqCyJuTd6LEGFttpmkYC91Hrk3s1GZFERMmUT01Xv/sQljjPlMRMsxO6WULwnb2D8FEs4j680wScjO5f3vzrlNJszESWq2LYXJgTzjZm56MCHf3zVBxH1r7ftU1splxxKYHEgoUUpTo+grEf303rPH5hxENJqDKQEJtko2q9zGeeycWy3JhpKhWT8+NM/sufIhBwKI+Mta+7pkfxKMtd8Qtdbcx4dUQZcFCQ2I6DcAnLUpf6YMPxhIDDOuxC4C6djoQUE6+tKpewWZ1wlRkq0qUhXptKTlzv93aI3jWmE0Fz2TeujpX73F9TaKy9CeMk8vZusfBnqZ1g5GqyIdJq+XrqNR5AahKr9CCcxGSwAAAABJRU5ErkJggg==)

GPU 识别检查
--------

​ 系统安装完成后，登陆虚拟机系统使用 lspci 命令检查 GPU 识别情况，如下表示添加的两块 GPU 识别正常。

​ 最后从 NVIDIA 官网下载对应的 GPU 驱动并安装，安装后建议打开 GPU 驱动`persistence mode`并配置开机自启动：

可能会遇到的问题
--------

### ESXi 安装时卡在 bnxtroce.v00

​ 出现该问题多为刻录 U 盘时选择的 U 盘格式有问题，建议使用本文介绍的 rufus 工具进行刻录，同时物理机引导模式选择 EFI 引导。若使用软碟通进行 U 盘刻录，可以将写入方式修改为 **USB-ZIP+ v2** 或者 **USB-HDD+ v2**。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/ESXi%E5%AE%89%E8%A3%85%E5%8D%A1%E4%BD%8F-20210118181122.png)

### 系统安装时找不到引导介质

​ 如下，系统安装时显示找不到引导介质，可以将`CD/DVD驱动器1`删除后重新添加，重新配置引导介质。

<img src="file:///E:\ 软件资料存储 \ qq\1037509307\Image\C2C]T0_J}$NN1C]Q60Y_]1G{}E.png"alt="img" />

<img src="[https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E6%97%A0%E5%BC%95%E5%AF%BC%E4%BB%8B%E8%B4%A8-20210118182049.png](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E6%97%A0%E5%BC%95%E5%AF%BC%E4%BB%8B%E8%B4%A8-20210118182049.png)" />

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/DVD%E9%A9%B1%E5%8A%A8%E5%99%A8-20210118181807.png)

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/%E6%B7%BB%E5%8A%A0DVD%E9%A9%B1%E5%8A%A8%E5%99%A8-20210118181907.png)

### ESXi 6.0 安装后无法通过浏览器进行管理

​ 添加网页管理 web client 的方式是：

​ ESXi 控制台界面，按 F2 进入系统配置，输入用户名 / 密码后，进入 troubleshooting options 中，按回车键打开 SSH。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/web1-20210119093430.png)

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/web2-20210119093542.png)

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/web3-20210119093633.png)

​ 使用 xshell 等终端 ssh IP，进入命令行窗口，然后通过 ssh 运行安装 web client 的命令：

安装完成后，可以使用浏览器打开 [http://IP/ui](http://ip/ui) 来进行网页管理。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/20210119105141-CLI.png)

### 浏览器管理界面密码输入正确但无法登录

​ 连接物理服务器按`F2`键进入 ESXi 控制台界面，进入`Troubleshooting Options`, 选择`Restart Managent Agents`。若仍然无效，可以先在控制界面修改登录密码后再执行此操作。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/20210119112814.png)

### 虚拟机无法开机，提示电源报错

​ 具体报错内容如下：无法打开虚拟机的电源，失败 - 模块 “DevicePowerOn” 打开电源失败。

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/20210119105659-%E6%89%93%E5%BC%80%E7%94%B5%E6%BA%90%E5%A4%B1%E8%B4%A5.png)

可能原因为：

*   ESXi 软件与 GPU 不兼容
*   ESXi 软件中未设置 MMIO 相关参数
    *   pciPassthru.use64bitMMIO="TRUE"
    *   pciPassthru.64bitMMIOSizeGB =“<n>”

### GPU 驱动安装报错

​ 虚拟机内 GPU 驱动安装失败，提示以下报错：

![](https://gitee.com/Gavin_zj/blog/raw/master/blog_img/20210119111514-GPU%E9%A9%B1%E5%8A%A8%E6%8A%A5%E9%94%99.png)

可能原因：

*   虚拟机操作系统引导方式为 BIOS，需要修改为 EFI；
*   ESXi 软件中未设置 MMIO 相关参数
    *   pciPassthru.use64bitMMIO="TRUE"
    *   pciPassthru.64bitMMIOSizeGB =“<n>”