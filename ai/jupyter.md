# Jupyter notebook/Lab

Jupyter Notebook是基于网页的用于交互计算的应用程序，其保存的是`.ipynb`的`JSON`格式文件，爱py牛逼！

Jupyter Notebook还有个“双胞胎”——Jupyter Lab。Jupyter Lab是基于Web的集成开发环境，可以把它当作进化版的Jupyter Notebook。使用Jupyter Lab可以同时在一个浏览器页面打开编辑多个Notebook、Ipython console和terminal终端，甚至可以使用Jupyter Lab连接Google Drive等服务。由于Jupyter Lab拥有模块化结构，提供更多类似IDE的体验，已经有越来越多的人从使用Jupyter Notebook转向使用Jupyter Lab。

## Jupyter Notebook/Lab安装

安装Jupyter Notebook：激活虚拟环境后，输入

```
conda install jupyter notebook
# pip install jupyter notebook
```

安装Jupyter Lab：激活虚拟环境后，输入

```
conda install -c conda-forge jupyterlab
# pip install jupyterlab
```

在终端输入指令打开

```
# 打开Jupyter Notebook
jupyter notebook
# 打开Jupyter Lab
jupyter lab
# 指定端口
jupyter notebook --port <port_number>
# 启动服务不打开浏览器
jupyter notebook --no-browser
```

## Jupyter Notebook/Lab配置

默认打开的是home目录下的文件，如果我们想要更改默认文件存放路径，该怎么办？

```
jupyter notebook --generate-config
```

输入这条命令用于查看配置文件的路径。

常规的情况下，Windows和Linux/macOS的配置文件所在路径和配置文件名如下所述：

- Windows系统的配置文件路径：`C:\Users\<user_name>\.jupyter\`
- Linux/macOS系统的配置文件路径：`/Users/<user_name>/.jupyter/` 或 `~/.jupyter/`
- 配置文件名：`jupyter_notebook_config.py`

进入配置文件后查找关键词“c.NotebookApp.notebook_dir”，添加如下所示：

```
c.NotebookApp.notebook_dir = 'D:\Projects\PYprojects'
```

这样每次打开都是指定的路径。

## 打通conda虚拟环境和Jupyter Notebook/Lab

Anaconda安装的虚拟环境和Jupyter Notebook运行需要的Kernel并不互通。那么我们该如何解决这个问题，并且如果我们想要切换内核（Change Kernel），该如何操作呢？

### 创建虚拟环境时添加ipykernel

例如在创建pytorch虚拟环境时直接加上参数ipykernel`conda create -n pytorch python=3.9 ipykernel`

### 创建虚拟环境后，再添加ipykernel

如果在创建虚拟环境时，没有使用参数ipykernel，可以通过install命名追加，命令格式`conda install -n 虚拟环境名字 ipykernel`

例如创建pytorch环境时，没有使用ipykernel：`conda create -n pytorch python=3.9`，再想添加ipykernel，可以使用下面的命令：`conda install -n pytorch  ipykernel`或者在环境中，使用`pip install ipykernel`

激活环境后，将虚拟环境加入 jupyterlab 的 kernel 中，格式为：`python -m ipykernel install --user --name 虚拟环境名字 --display-name "虚拟环境名字"`

## 插件系统

激活环境，

```
pip install jupyter_contrib_nbextensions
# 将插件添加到工具栏
jupyter contrib nbextension install
```

然后打开Jupyter Notebook，就是打开的默认主页，点击Nbextensions，取消勾选`disable configuration for nbextensions without explicit compatibility`

推荐插件：

- Execute Time：可以显示执行一个Cell要花费多少时间
- Hinterland：提供代码补全功能

安装方式：

1. Nbextensions里搜索，给你搜到的插件名前面的选项框打个√，或者点击下面介绍里的Enable按钮。
2. 命令行安装：`jupyter labextension install jupyterlab-execute-time`



## 基本操作

| 在编辑模式下的操作           |                                        |
| ---------------------------- | -------------------------------------- |
| Tab                          | 补全代码 或 缩进                       |
| Shift + Tab                  | 提示                                   |
| Ctrl + ] 或 [                | 添加 / 删除缩进                        |
| Ctrl + Z                     | 撤销                                   |
| Ctrl + Y 或 Ctrl + Shift + Z | 恢复                                   |
| Ctrl + S                     | 保存                                   |
| Ctrl + Enter                 | 运行本单元格                           |
| Shift + Enter                | 运行本单元格，并转到下一个单元格       |
| Alt + Enter                  | 运行本单元格，并在下方插入新单元格     |
| Esc                          | 编辑模式 to 命令模式                   |
| 在命令模式下                 |                                        |
| A                            | 在上方插入新单元格                     |
| B                            | 在下方插入新单元格                     |
| C                            | 复制选中的单元格                       |
| D（连续按两次）              | 删除单元格                             |
| L                            | 添加行号                               |
| V / Shift + V                | 粘贴到下方单元格 / 上方单元格          |
| X                            | 剪切（注：复制/剪切/粘贴不能跨笔记本） |
| Y                            | 将单元格改为 代码格式                  |
| M                            | 改为 markdown格式                      |

如果要使用一些Bash命令行，需要在前面添加“!”，例如解压文件`!unzip (压缩包所在路径) -d (解压路径)`，`!ls | wc -l`当前文件夹下文件的数量，`!python demo.py`，`!pip install ...`

魔术方法：

timeit，运行多次的平均值用来评估运行效率，两个%表示多行，且 %%timeie 必须放在第一行

```
# 一行
%timeit choose_sort(lst)

# 多行，从下一行开始测试下面所有单元格的执行时间
%%timeit

choose_sort(lst)
sorted(lst)
```

占用内存要用到第三方库 memory_profiler，然后在单元格中导入 

```python
%load_ext memory_profiler
```

在需要测量内存的代码单元格上方加上 `%%memit`魔法命令 或者，可以使用 `%memit`魔法命令来测量单个代码行的内存使用情况

测量结果将包含每个代码行的内存使用情况，以及代码运行结束时的峰值内存使用量。

请注意，`memory_profiler`在测量内存使用时会对代码的执行速度产生一些影响，因此在测量大型代码或长时间运行的代码时可能会导致运行时间延长。
