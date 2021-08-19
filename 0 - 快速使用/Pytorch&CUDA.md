[一系列环境搭建](https://blog.csdn.net/weixin_41803874/article/details/91913063)

> conda可用一条install指令一次性为我们一并安装PyTorch + torchvision + cuda，
>
> 具体指令可以在pytorch官网（https://pytorch.org/）

[Ubuntu16.04服务器上用conda安装PyTorch、torchvision、cuda](https://blog.csdn.net/Trasper1/article/details/100039450)

1. 查询pytorch版本

```
torch       1.9.0
torchvision    0.10.0
```

```python
python
>> import torch
>> Print(torch.__version__)
```

2. 查询CUDA版本

> nvcc -V

> nvidia -smi

nvcc -v 和 nvidia-smi 两个对应的CUDA版本并不一样：CUDA 有两种API，分别是 运行时 API 和 驱动API，即所谓的 Runtime API 与 Driver API。 nvidia-smi 的结果除了有 GPU 驱动版本型号，还有 CUDA Driver API的型号，这里是 10.0。而nvcc的结果是对应 CUDA Runtime API。
而我们安装的时候，要和nvcc的保持一致。
ref: [Cudatoolkit版本的选择](https://blog.csdn.net/weixin_41515338/article/details/109207276)

3. 如何读懂 **nvidia -smi**

ref：[CUDA之nvidia-smi命令详解](https://blog.csdn.net/u013066730/article/details/84831552)

![image-20210811113851228](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/08_11_image-20210811113851228.png)

**第一栏的Fan：N/A是风扇转速，从0到100%之间变动。有的设备不会返回转速，因为它不依赖风扇冷却而是通过其他外设保持低温。**
**第二栏的Temp：是温度，单位摄氏度。**
第三栏的Perf：是性能状态，从P0到P12，P0表示最大性能，P12表示状态最小性能。
第四栏下方的Pwr：是能耗，上方的Persistence-M：是持续模式的状态，持续模式虽然耗能大，但是在新的GPU应用启动时，花费的时间更少，这里显示的是off的状态。
第五栏的Bus-Id是涉及GPU总线的东西，domain?device.function

**第六栏的Disp.A是Display Active，表示GPU的显示是否初始化。**
**第五第六栏下方的Memory Usage是显存使用率。**

第七栏是浮动的GPU利用率。
第八栏上方是关于ECC的东西。
第八栏下方Compute M是计算模式。

4. 查询cudatolkit版本

> conda list

![image-20210811114922286](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/08_11_image-20210811114922286.png)