> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/asla_1/article/details/105209424)

主要原因是因为使用了 proxy 代理，需要关闭代理。

  git config --global http.proxy  // 查看代理

 结果为：localhost:1080 

  git config --global --unset http.proxy  // 不设置代理

  再拉取就没有问题了。

