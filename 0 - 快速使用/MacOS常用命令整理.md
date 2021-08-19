## 环境变量

> 原文地址 [www.jianshu.com](https://www.jianshu.com/p/07494f169380)

*   打开环境变量配置

1. Mac 的环境变量配置保存在用户目录wangyangfan下的**.bash_profile**中，是共享的目录同级，下载目录的上一级。

2. 显示隐藏文件:如果你想显示隐藏的文件夹，可以使用快捷键：⌘+ ⇧+.三个按键，再按下去就是隐藏了。

在该目录下，打开终端，输入如下命令，就可以打开该文件：

```
open .bash_profile
```

*   增加环境变量

1.  例如 Android 开发环境，需要在别的目录调用 adb 系列命令，就要添加 Android 目录，默认目录在 Users 目录下，后面的 lizhi 是电脑名，根据你的配置来改喔

```
export ANDROID_HOME=/Users/lizhi/Library/Android/sdk
export PATH=${PATH}:${ANDROID_HOME}/tools
export PATH=${PATH}:${ANDROID_HOME}/platform-tools
```

2.  Python3 开发环境配置，注意每个人安装的版本不一样，目录文件夹就不一样，后面的 3.7.7 要根据安装的具体版本改喔

```
export PATH=${PATH}:/usr/local/Cellar/python/3.7.7/bin
alias python="/usr/local/Cellar/python/3.7.7/bin/python3"
alias pip="/usr/local/Cellar/python/3.7.7/bin/pip3"
```

*   保存环境变量

改完文件，保存，并不代表应用，需要使用如下命令

```
source ./.bash_profile
```

*   验证

应用完毕后，在终端输入 adb，能出现 adb 命令的帮助。或者 python --version，能出现 python3 版本提示，就代表配置成功了。



## Finder分栏模式的宽度设置

>  原文地址 [blog.csdn.net](https://blog.csdn.net/gnail_oug/article/details/79848188)

问题描述：

默认情况下，访达 (Finder) 分栏显示 - 分栏的宽度很窄，文件名长的时候显示不全，很是不方便。

解决方法：  

1、打开 Finder，设置以分栏方式显示  

2、按住`option`键，用鼠标拖动调整分栏的宽度。这样再次打开 Finder 时分栏默认宽度就是刚调整的宽度。  

3、验证，按住`option`键，右键 Finder 图标–> 重新开启，就会发现分栏宽度是调整后的宽度。



## 快捷键

1. ”您不能使用'.'开头的名称 ，因为这些名称已经被系统预留，请选择其他名称“ > Finder显示隐藏文件：⌘+ ⇧+.三个按键
2. 重启Finder：按住`option`键，右键 Finder 图标–> 重新开启

### 移动光标+选中文字

跳到本行开头 – Command + 左方向键← 
跳到本行末尾 – Command + 右方向键→ 
跳到当前单词的开头 – Option + 左方向键← 
跳到当前单词的末尾 – Option + 右方向键→ 
跳到整个文档的开头 – Command + 上方向键↑ 
跳到整个文档的末尾 – Command + 下方向键↓



选中当前位置到本行开头的文字 – Shift + Command + 左方向键← 
选中当前位置到本行末尾的文字 – Shift + Command + 左方向键→ 
选中当前位置到所在单词开头的文字 – Shift + Option + 左方向键← 
选中当前位置到所在单词末尾的文字 – Shift + Option + 右方向键→ 
选中当前位置到整个文档开头的文字 – Shift + Command + 上方向键↑ 
选中当前位置到整个文档末尾的文字 – Shift + Command + 下方向键↓