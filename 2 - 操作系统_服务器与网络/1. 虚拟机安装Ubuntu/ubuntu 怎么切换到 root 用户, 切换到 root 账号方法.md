> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [jingyan.baidu.com](https://jingyan.baidu.com/article/fd8044fa1e74035031137ae0.html)

ubuntu 怎么切换到 root 用户，我们都知道使用 su root 命令，去切换到 root 权限，此时会提示输入密码，可是怎么也输不对，提示 “Authentication failure”，

此时有两种情况一个是真的是密码错了，另一种就是刚安装好的 Linux 系统，没有给 root 设置密码。

通过下文就可以解决这两个问题。

[](javascript:;)方法 / 步骤
-----------------------

1.  1
    
    打开 Ubuntu，输入命令：su root，回车提示输入密码，怎么输入都不对
    
    ![](https://exp-picture.cdn.bcebos.com/2e223d85e036e2915dc19255b2723d03baea5b82.jpg?x-bce-process=image%2Fresize%2Cm_lfit%2Cw_500%2Climit_1%2Fquality%2Cq_80)
2.  2
    
    给 root 用户设置密码：
    
    命令：sudo passwd root
    
    输入密码，并确认密码。
    
    ![](https://exp-picture.cdn.bcebos.com/22c4fe36e29147e828c998c7b603bbea3f865882.jpg?x-bce-process=image%2Fresize%2Cm_lfit%2Cw_500%2Climit_1%2Fquality%2Cq_80)
3.  3
    
    重新输入命令：su root
    
    然后输入密码：
    
    发现可以切换到 root 权限了。
    
    ![](https://exp-picture.cdn.bcebos.com/e177fc9147e833e0255b9cb630ea3e8631485982.jpg?x-bce-process=image%2Fresize%2Cm_lfit%2Cw_500%2Climit_1%2Fquality%2Cq_80)
4.  4
    
    使用 su xyx 命令，切换到普通用户。
    
    ![](https://exp-picture.cdn.bcebos.com/e3d059e833e03972202a1a5fb586304860435682.jpg?x-bce-process=image%2Fresize%2Cm_lfit%2Cw_500%2Climit_1%2Fquality%2Cq_80)END

经验内容仅供参考，如果您需解决具体问题 (尤其法律、医学等领域)，建议您详细咨询相关领域专业人士。_作者声明：_本篇经验系本人依照真实经历原创，未经许可，谢绝转载。展开阅读全部