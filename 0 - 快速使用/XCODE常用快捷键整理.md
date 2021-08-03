Xcode自带Git的使用｜[https://u.nu/1vtp](https://u.nu/1vtp)

1. Debug -调试

* 设置断点

![图片](https://uploader.shimo.im/f/0Eg4WbeNVasb0xWz.png!thumbnail?fileGuid=YCT8G3qxX9KVRvGY)

* 逐语句与逐过程

逐语句是**进入**函数内部，进行单步调试。

逐过程就是把一个函数当成一条语句，**不进入**函数内部。

* 单步调试step into/step out/step over区别

step into：单步执行，遇到子函数就**进入**并且继续单步执行（简而言之，进入子函数）；

step out：当单步执行到子函数内时，用step out就可以执行完子函数余下部分，并返回到上一层函数。

step over：在单步执行时，在函数内遇到子函数时**不会进入**子函数内单步执行，而是将子函数整个执行完再停止，把子函数整个作为一步。在不存在子函数的情况下是和step into效果一样的（越过子函数，但子函数会执行）。

2. 编译并发布released 版本

![图片](https://uploader.shimo.im/f/fJlm16KcTlJ4zdmq.png!thumbnail?fileGuid=YCT8G3qxX9KVRvGY)

![图片](https://uploader.shimo.im/f/9ow7tSruXqnJTcGe.png!thumbnail?fileGuid=YCT8G3qxX9KVRvGY)



3. 更改 Xcode中所用的编译器类型：

![图片](https://uploader.shimo.im/f/1IzoQJbikx2gmAP1.png!thumbnail?fileGuid=YCT8G3qxX9KVRvGY)

4. 运行到当前行：

![图片](https://uploader.shimo.im/f/OPZItxhHoVG3O5zX.png!thumbnail?fileGuid=YCT8G3qxX9KVRvGY)

5. 代码折叠

在Xcode菜单里选择Preference——Text Editing，你会发现里面有一个“code folding ribbon”，勾选它就能恢复代码折叠功能了。

6. format

选中代码，control+i

7. 批量重命名变量：

选中变量名-> refactor -> rename

![图片](https://uploader.shimo.im/f/ZB0ShzfPVMXwtexz.png!thumbnail?fileGuid=YCT8G3qxX9KVRvGY)




![图片](https://uploader.shimo.im/f/vQZa3N5t2GuC94Qv.png!thumbnail?fileGuid=YCT8G3qxX9KVRvGY)



8. 恢复已关闭的标签页

Crtl + Shift + T

9. 真正的全屏

Shift+cmd+F

![图片](https://uploader.shimo.im/f/MjHTzEd7VH3gltpi.png!thumbnail?fileGuid=YCT8G3qxX9KVRvGY)

10. 上下切换标签页

下一页：^ + ->

上一页：^ + shift + ->






