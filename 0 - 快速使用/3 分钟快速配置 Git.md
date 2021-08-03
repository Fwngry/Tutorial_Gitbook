>  原文地址 [blog.csdn.net](https://blog.csdn.net/reykou/article/details/104773190/)

**目录**

[1 在 Github 上创建远程仓](#t1)

[2 在 VSCode 上初始化本地仓](#t2)

[3 在 VSCode 上绑定远程仓](#t3) 

[4 在 VSCode 上拉取远程仓内容](#t4)

[5 在 VSCode 上新建文件并提交到本地仓](#t5)

[6 在 VSCode 上将本地仓更改推送到 Github 远程仓](#t6)

```bash
# 设置邮箱和用户名
git config --global user.email "wyfsgm@gmail.com"
git config --global user.name "Fwngry"  

#改为国内网络
git init
git remote add origin https://github.com/Fwngry/DeepSort_DataUtils.git

# 先进行拉取
git pull origin master
git add --all
git commit -m "commit Message"

# git原生分支master，github原生分支main，改名后直接提到main中
git branch -M main
git push -u origin master
```

https://www.bilibili.com/video/av95681488/)

### 1 在 Github 上创建远程仓

创建时、可直接选择 ☑️ "Initialize this repository with a README"、初始化仓库。![](https://img-blog.csdnimg.cn/20200310142106333.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

1.  ![](https://img-blog.csdnimg.cn/20200310142023519.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

此步骤相当于创建空的远程仓后在终端执行以下几句话：

```
echo "# GIT" >> README.md
git init
git add README.md
git commit -m "first commit"
```

### 2 在 VSCode 上初始化本地仓

在本地电脑上创建空文件夹、如 GitTestForMac。

![](https://img-blog.csdnimg.cn/20200310142828303.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

初始化本地 Git 存储库的工作区文件夹、点击『源代码管理』右上角『+』、在弹出对话框中选择默认的 GitTestForMac 

![](https://img-blog.csdnimg.cn/20200310143509301.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20200310143631115.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

### 3 在 VSCode 上绑定远程仓 

按 Shift+Command+P 、在弹出框中选择 『Git 添加远程仓』、远程仓库名称填写『origin』、远程仓的 URL 输入在 GITHUB 创建远程仓的 git 地址。

![](https://img-blog.csdnimg.cn/20200310143714188.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20200310145608820.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/2020031014574218.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20200310143758197.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

此步骤相当于用终端输入：

```
git remote add origin https://github.com/Reykou/GitTest.git
```

> _如不执行以上动作会提示错误：『存储库未配置任何要推送到的远程存储库。』_
> 
>  ![](https://img-blog.csdnimg.cn/2020031222370091.png)

### 4 在 VSCode 上拉取远程仓内容

点击『源代码管理器: Git』界面中的右上角的『...』、选择『拉取自...』、在弹出菜单中依次选择『origin』,『origin/master』。拉取成功后完成后即可在文件栏中看到文件 README.md

_常见问题：在第二个弹出菜单中没有『origin/master』选项怎么办？_

_稍等 2-3 分钟、重试即可出现。_ 

![](https://img-blog.csdnimg.cn/20200310143902635.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20200310143923981.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20200310143942157.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/202003101440035.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

相当于在终端输入：

```
git pull origin master
```

### 5 在 VSCode 上新建文件并提交到本地仓

添加文件、在『源码管理器中』选择『√』，提交到本地仓。

![](https://img-blog.csdnimg.cn/20200310144025241.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

### 6 在 VSCode 上将本地仓更改推送到 Github 远程仓

在『源码管理器中』选择『推送到...』、在弹出框中选择分支『origin』。推送成功后、刷新 GitHut 工程界面、即可看到提交的新文件。

![](https://img-blog.csdnimg.cn/20200310144100371.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20200310144123101.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20200310142710971.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM2NDUyMTk=,size_16,color_FFFFFF,t_70)

此步骤相当于在终端输入：

```
git push -u origin master
```

 GIT 基础知识备忘录
------------

> 三种状态
> 
>  Git 有三种状态，你的文件可能处于其中之一： **已提交（committed）**、**已修改（modified）** 和 **已暂存（staged）**。
> 
> *   已修改表示修改了文件，但还没保存到数据库中。
>     
> *   已暂存表示对一个已修改文件的当前版本做了标记，使之包含在下次提交的快照中。
>     
> *   已提交表示数据已经安全地保存在本地数据库中。
>     
> 
> 这会让我们的 Git 项目拥有三个阶段：工作区、暂存区以及 Git 目录。 This leads us to the three main sections of a Git project: the working tree, the staging area, and the Git directory.
> 
> ![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9naXQtc2NtLmNvbS9ib29rL2VuL3YyL2ltYWdlcy9hcmVhcy5wbmc?x-oss-process=image/format,png)
> 
> 基本的 Git 工作流程如下：
> 
> 1.  在工作区中修改文件。
>     
> 2.  将你想要下次提交的更改选择性地暂存，这样只会将更改的部分添加到暂存区。
>     
> 3.  提交更新，找到暂存区的文件，将快照永久性存储到 Git 目录。

参考：[GIT 官方文档](https://git-scm.com/book/zh/v2/%E8%B5%B7%E6%AD%A5-Git-%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F)