# VMWare - vSphere

VMWare - vSphere：虚拟化服务器管理工具

VMWare - vSphere -DOC：https://docs.vmware.com/cn/VMware-vSphere/index.html

VMWare - vSphere = ESXi + vCenter Server

<img src="https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/04_27_image-20210427204550673.png" alt="image-20210427204550673" style="zoom:50%;" />

以下操作都是在服务器端进行：进入服务器管理平台Esxi - 释放 - 整合 - 虚拟机安装linux

1. 进入ESXi：

https://193.168.1.119/ui/#/host/vms/2
账号：root；密码：wave@123

2. 释放虚拟机：先释放后创建，整合资源

可释放的机器：
CentOS_7.6.1810_compile_02
CentOS_7.6.1810_single_01
CentOS_7.6.1810_integrate_02
CentOS_7.6.1810_functional02

3. 创建新虚拟机

选择内存32G/核数16核/硬盘容量/CDDVD - 加载Ubuntu - ISO（清华镜像）

operating system not found - 创建虚拟机后，要安装光盘才行.

4. VMWare tools

5. ID&Password

   <img src="https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/04_28_04_28_Apr_27_21.png" alt="password" style="zoom: 33%;" />

<img src="https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/04_28_04_28_password.png" alt="password" style="zoom:33%;" />