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