# PyCharm 详细使用指南

## 一、基础篇

### 1. 运行代码

- 在 Mac 系统中使用快捷键 **Ctrl+Shift+R**，在 Windows 或 Linux 系统中，使用快捷键 Ctrl+Shift+F10。

- **右键单击背景，从菜单中选择“Run 『guess_game』”**。

- 由于该程序具备__main__ 从句，你可以点击__main__ 从句左侧的绿色小箭头，选择「Run 『guess_game』」

### 2. Debug

- 在 Mac 系统中使用 **Ctrl+Shift+D** 键，在 Windows 或 Linux 系统中使用 Shift+Alt+F9 键。

- **右键单击背景，选择「Debug 『guess_game』」**。

- 点击__main__从句左侧的绿色小箭头，选择「Debug 『guess_game』」。

1. Debugger 窗口查看变量
2. console 窗口进行交互
3. F8继续下一行；F7进入函数

### 3. 测试单元

PyCharm 使得为已有代码创建测试变得轻而易举。可以对某文件的效果单独进行测试，无非就是免去了自己写一些调用的程序，我觉得也就还好吧~

#### 设置

打开 Settings/Preferences → Tools → Python Integrated Tools 设置对话框。

在默认测试运行器字段中选择 pytest。

点击 OK 保存该设置。

#### 生成测试单元

注意，光标放在函数or类定义的位置。

- 在 Mac 系统中使用 Shift+Cmd+T 键，在 Windows 或 Linux 系统中使用 Ctrl+Shift+T。
- **右键单击该类的背景，选择「Go To and Test」**。
- 在主菜单中，选择 Navigate → Test。

点击Create New Test，选择需要测试的单元。Target directory、Test file name 和 Test class name 这三项均保留默认设置，生成了对应的测试文件。

#### 修改测试文件

根据对被测试单元的调用需求，改动测试文件

#### 进行测试

- 在 Mac 系统中使用 Ctrl+R 键，在 Windows 或 Linux 系统中使用 Shift+F10 键。

- 右键单击背景，选择「Run 『Unittests for test_calculator.py』」。
- **点击测试类名称左侧的绿色小箭头，选择「Run 『Unittests for test_calculator.py』」**

![image-20210820211726956](/Users/wangyangfan/Library/Application Support/typora-user-images/image-20210820211726956.png)

测试结果在屏幕下端显示

### 4. 搜索与导航

#### 搜索

- 当前文件+搜索代码段：在 Mac 系统中使用 Cmd+F 键，在 Windows 或 Linux 系统中使用 Ctrl+F 键。
- 整个项目+搜索代码段：在 Mac 系统中使用 Cmd+Shift+F 键，在 Windows 或 Linux 系统中使用 Ctrl+Shift+F 键。
- 搜索类：在 Mac 系统中使用 Cmd+O 键，在 Windows 或 Linux 系统中使用 Ctrl+N 键。
- 搜索文件：在 Mac 系统中使用 Cmd+Shift+O 键，在 Windows 或 Linux 系统中使用 Ctrl+Shift+N 键。
- 如果你不知道要搜索的是文件、类还是代码段，则搜索全部：按两次 Shift 键。

#### 导航

- 前往变量的声明：在 Mac 系统中使用 Cmd 键，在 Windows 或 Linux 系统中使用 Ctrl 键，然后单击变量。
- 寻找类、方法或文件的用法：使用 Alt+F7 键。
- 多次跳转后在导航历史中前进和后退：在 Mac 系统中使用 Cmd+[ / Cmd+] 键，在 Windows 或 Linux 系统中使用 Ctrl+Alt+Left / Ctrl+Alt+Right 键。

更多细节，参见官方文档：https://www.jetbrains.com/help/pycharm/tutorial-exploring-navigation-and-search.html。



----

## 二、问题篇

### 1. 导包提示 unresolved reference

>  原文地址 [blog.csdn.net](https://blog.csdn.net/sinat_34104446/article/details/80951611)

描述：模块部分，写一个外部模块导入的时候居然提示 unresolved reference，如下，程序可以正常运行，但是就是提示包部分红色，看着特别不美观，下面是解决办法

解决：https://u.nu/4saec

       1. 进入PyCharm->Preferences->Build,Excution,Deployment->Console->Python Console勾选上Add source roots to PYTHONPATH;
       2. 进入PyCharm->Preferences->Project->Project Structure,通过选中某一目录右键添加sources;
          3. 点击Apply和OK即可.

###  2. python解释器 - no python interpreter configured

1. File–>Setting–>Project，这时候看到选中栏显示的是 No interpreter
2. Terminal：which python 3,把结果填入刚才的地址栏

### 3. 显示当前 python 文件下的函数和类的列表

显示每个 py 文件里面的类和方法，方便快速跳转。

方法一：  

左侧 project 工具栏窗口顶部那个齿轮有个 show member 选项，默认是不开的，勾选后 py 文件会显示内部定义的 class 和 method ，每个文件可以自由选择折叠还是展开。

![](https://img-blog.csdnimg.cn/20190609162135977.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FuYWtpbjYxNzQ=,size_16,color_FFFFFF,t_70)

方法二：

只能对选择某个文件来展开

![](https://img-blog.csdnimg.cn/20190609161858371.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FuYWtpbjYxNzQ=,size_16,color_FFFFFF,t_70)



## 4. 查看python的变量类型和变量内容 - debug

首先，在程序的某一处添加断点，点击行号右边部分红处，如下图所示：

![](https://img-blog.csdn.net/20161206114335761?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)  

添加断点后，选择 debug 程序，快捷键在 pycharm 的右上角。

![](https://img-blog.csdn.net/20161206114645119?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)  

debug 过程中，pycharm 的下方工作区域内会相应显示：

![](https://img-blog.csdn.net/20161206115053780?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)  

Variables 窗口中的变量可以右击，Add to Watches，然后在 Watches 窗口中可以看到所选数据的具体信息，包括数值。熟练利用还是比较方便的。

---

## 三、需求篇

### 场景一：使用Pycharm远程连接服务器（传输同步文件、代码调试）

Connection refused 排查过程https://www.cnblogs.com/feshfans/p/9663291.html

PyCharm远程调试Python的三种方式
https://blog.csdn.net/dkjkls/article/details/89054595

Python远程调试图文教程（一）之Pycharm Remote Debug 
https://blog.51cto.com/u_15009285/

新建端口：http://www.xitongcheng.com/jiaocheng/win10_article_12908.html

#### 1. 前置动作

macOS 2021.2 Pycharm IDE默认没找到Deployment选项，[解决](https://blog.csdn.net/yuebowhu/article/details/104056545)：

> Preference --> Appearance --> Menus and Toolbars 
>
> 点开Tools文件夹，选中Tools中的一个子文件夹（Deployment就放在了这个工具后面了）
>
> 点击上面的+号，选择choose action to add
>
> 搜索deployment，找到Deployment文件夹，选中此文件夹，
>
> 点击Apply
>
> 点击OK

#### 2. 远程连接

#### 3. Mapping同步文件夹

#### 4. ssh python解释器

#### 5. Run config

