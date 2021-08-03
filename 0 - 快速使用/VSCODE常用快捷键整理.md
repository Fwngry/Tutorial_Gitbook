## 代码折叠 fold

- 操作所有代码块：
  - 折叠所有 `Ctrl+K+0`
  - 展开所有 `Ctrl+K+J`
  
- 操作`代码块`：
  - 折叠 `Ctrl+Shift+[`
  - 展开 `Ctrl+Shift+]`
  
  ## 打开Setting.json
  
  在VS Code中键入ctrl+shift+P全局快捷键，打开命令搜索窗，输入settings.json即可打开首选项。

## 外部引用问题

[www.pianshen.com](https://www.pianshen.com/article/84501691762/)

问题分析：

1. 外部引用无法“转到定义”
2. 报错：unresolved reference

解决思路：

问题根源在于，VSCODE未能识别到外部引用目录。因此需要为 launch.json添加配置，同时在项目根文件夹中新建.env文件并写明对应的外部库路径，一箭双雕解决“转到引用”和“外部引用”无法识别的问题。

1. 为 launch.json添加配置

   launch.json的位置：${ProjectFolds} -.vscode - launch.json。launch.json属于项目，用于启动和调试。

   <img src="https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/04_26_image-20210426100641865.png" alt="image-20210426100641865" style="zoom:50%;" />

   ```json
   "env": {"PYTHONPATH":"${workspaceRoot}"},
   "envFile": "${workspaceRoot}/.env"
   ```

   2. .env添加路径

   Command+Shift+.：显示隐藏文件

   新建.env：

   ```
   PYTHONPATH=./lib
   ```

   

## Pylance - 插件

支持 "转到定义" F12 功能，还讲源码中的包名和类名的关键字进行颜色区分显示，真的是实力与颜值俱在！

当然，此时已自动将 settings.json 中 python 语言服务器设置为 Pylance：

```
"python.languageServer": "Pylance"
```

- VS Code 中"转到定义"功能，核心是受 settings.json 中的 python.languageServer 参数控制，该参数合法取值有 Jedi、Microsoft 和 None，安装 Pylance 插件后还支持 Pylance。当设置为 Microsoft 和 None 时，无法实现转到定义，而设置 Jedi 和 Pylance 时可以。
- VS Code 中搭建 Python 环境，建议安装两个插件：即 Python+Pylance，其中前者是 VS Code 支持 Python 编译的前提，后者是基于 Python 的扩展，支持自动补全、参数提示、转到定义等多项功能改进。