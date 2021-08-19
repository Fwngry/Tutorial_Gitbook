##  导包提示 unresolved reference

>  原文地址 [blog.csdn.net](https://blog.csdn.net/sinat_34104446/article/details/80951611)

描述：模块部分，写一个外部模块导入的时候居然提示 unresolved reference，如下，程序可以正常运行，但是就是提示包部分红色，看着特别不美观，下面是解决办法

解决：https://u.nu/4saec

       1. 进入PyCharm->Preferences->Build,Excution,Deployment->Console->Python Console勾选上Add source roots to PYTHONPATH;
2. 进入PyCharm->Preferences->Project->Project Structure,通过选中某一目录右键添加sources;
    3. 点击Apply和OK即可.

##  no python interpreter configured

1. File–>Setting–>Project，这时候看到选中栏显示的是 No interpreter
2. Terminal：which python 3,把结果填入刚才的地址栏



## 显示当前 python 文件下的函数和类的列表

显示每个 py 文件里面的类和方法，方便快速跳转。

方法一：  

左侧 project 工具栏窗口顶部那个齿轮有个 show member 选项，默认是不开的，勾选后 py 文件会显示内部定义的 class 和 method ，每个文件可以自由选择折叠还是展开。

![](https://img-blog.csdnimg.cn/20190609162135977.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FuYWtpbjYxNzQ=,size_16,color_FFFFFF,t_70)

方法二：

只能对选择某个文件来展开

![](https://img-blog.csdnimg.cn/20190609161858371.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FuYWtpbjYxNzQ=,size_16,color_FFFFFF,t_70)



## 查看python的变量类型和变量内容 - debug

用过 Matlab 的同学基本都知道，程序里面的变量内容可以很方便的查看到，但 python 确没这么方便，对于做数据处理的很不方便，其实不是没有这个功能，只是没有发现而已，今天整理一下供大家相互学习。

首先，在程序的某一处添加断点，点击行号右边部分红处，如下图所示：

![](https://img-blog.csdn.net/20161206114335761?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)  

添加断点后，选择 debug 程序，快捷键在 pycharm 的右上角。

![](https://img-blog.csdn.net/20161206114645119?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)  

debug 过程中，pycharm 的下方工作区域内会相应显示：

![](https://img-blog.csdn.net/20161206115053780?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)  

Variables 窗口中的变量可以右击，Add to Watches，然后在 Watches 窗口中可以看到所选数据的具体信息，包括数值。熟练利用还是比较方便的。