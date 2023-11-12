# DCU使用教程

## 查看DCU参数信息

如果使用`rocminfo`能展现DCU卡、CPU的一些硬件配置，参数等等。

使用`rocminfo | grep Z100L`，grep后面是卡的名称，使用这个命令，我认为最大的作用就是看这个机器有**多少张**卡。

## 查看DCU运行状态

`rocm-smi`，类似英伟达的`nvidia-smi`

参数解释：

1. DCU —— Id
2. Temp —— 温度
3. AvgPwr —— 功耗（实际是平均，但称为功耗即可，官方都这么叫）
4. Pwrcap——最大功耗（容量）
5. VRAM —— 显存占用
6. DCU% —— 显存利用率

## Pytorch部署

下载地址：https://developer.hpccube.com/tool/

AI生态包->pytorch

### 验证环境：

`python3 -c "import torch;print(torch.cuda.is_available())"`

`python3 -c "import torch;print(torch.__version__)"`

### 单机单卡

`HIP_VISIBLE_DEVICES=0`:单卡环境变量：指定第0张DCU卡

```shell
HIP_VISIBLE_DEVICES=0 python3 main_acc.py --batch-size=64 --arch=resnet50 -j 6 --epochs=1 --save-path=/path/to/{save_model_dir} /path/to/{ImageNet_pytorch_data_dir}/
```

--arch 网络模型，和DCU官方提供的main_acc.py有关

### 单机多卡

1. 单节点多进程分布式：`python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of your training script)`
2. 多节点多进程分布式（2节点为例）：

```
Node 1:
python -m torch.distributed.launch
--nproc_per_node=NUM_GPUS_YOU_HAVE --nnodes=2 -- node_rank=0 --master_addr="192.168.1.1" --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of your training script)

Node 2:
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of your training script)
```

注意：nproc_per_node代表DCU 0 到DCU（(nproc_per_node - 1），在多节点时候注意node_rank。

3. `mpirun --allow-run-as-root --bind-to none -np 4 single_process.sh localhost resnet50 64`

## DCU性能分析

### hipprof介绍

hipprof是DCU Toolkit的性能分析工具，主要功能是根据用户的指令选项，启动待进行性能分析的应用程序，抓取性能分析数据，最终对性能分析数据进行存储、导出和展示。hipprof的基本用法是hipprof加指令选项加程序启动命令，在正确加载DTK环境后，可以使用hipprof -h查看hipprof的帮助信息。

| 参数           | 主要参数释义                                                 |
| -------------- | ------------------------------------------------------------ |
| -d             | 性能分析工具产生的数据文件的存储目录，默认为当前目录         |
| -o             | 输出文件名称，输出文件为CSV文件，默认名称为.csv              |
| --hiptx-trace  | 开启hiptx性能分析，结合roctx接口，做应用性能分析             |
| --hip-trace    | hip trace性能分析类型，hipprof默认分析hip trace，这个指令可以不加 |
| --mpi-trace    | 开启mpi程序的trace性能分析，支持mpi多节点多进程              |
| --db           | 命令后加db文件，可以重新生成json可视化文件                   |
| --db-merge     | 命令后加目录路径，可以合并目录下的db文件，并生成合并后的数据库文件和前端脚本 |
| --index-range  | 后面加start和end的hip api顺序号，用冒号分开，可以做某一范围性能分析 |
| --group-stream | 默认情况下生成的前端脚本memcpy和kernel是分开显示的，加上此命令，可以使copy和kernel统一按照stream分组显示 |

#### hipprof单进程

1. 编译：`hipcc multiStream.cpp -o multiStream`
2. 运行：`hipprof --hip-trace ./multiStream`

除程序原有的输出外，我们还可以看到hipprof已经为程序输出了统计信息，按照总消耗时间从大到小的顺序，依次展示HIP Runtime API的调用名称、次数、总消耗时间和平均消耗时间。运行完成后当前目录生成默认结果文件，包括：

- hip-prof-18861.db

- result_18861.kernel.csv

- result_18861.hiptrace.csv

- result_18861_copy.json

  如果使用指令`hipprof --hip-trace --group-stream ./multiStream` 会生成stream.json 文件：

- result_26313_stream.json

  其中数字是本次执行hipprof的进程号，每次执行hipprof的时进程号不一定相同，db结尾的文件是**sqlite3**数据库文件，里面存放了全部的性能分析数据，csv文件可以用excel打开，是hip api的统计数据，包括调用次数和耗时统计情况，json文件是可以下载下来进行可视化的文件。

获取json文件后，打开chrome浏览器，地址栏输入: `chrome://tracing` or `edge://tracing/`，点击load按钮，选择下载好的json文件，就可以看到可视化的性能数据，用键盘的A、D、W、S按键可以左右移动，点击悬浮工具栏的“↓”，按住左键上下移动，可以放大和缩小显示区域，可视化后页面的最左侧是目录结构，中间是对应的性能时间线。

目录结构中：

- Runtime API

   显示hip API的性能情况，时间线反映了API的调用顺序以及占用线程时长。

- Memory Copy on Device 0

   指的是在device Id为0的设备上发生memcpy的情况，下边的H2D和D2H是 memcpy的类型方向，比如hipDeviceToHost、hipHostToDevice。

- Compute on Device 0 

  指的是在device Id是0的设备上发生的核函数的运行情况，下边的数字分别代表不同的stream，0代表的是默认stream。
  由于核函数或者memcpy和hip应用之间存在异步执行的情况，在点击选择核函数后，页面下部会显示一些详细信息及事件信息，点击Preceding events后边的链接，可以显示出对应hip接口的时间情况。

### torchprof介绍

torchprof用于Pytorch模型的逐层分析的最小依赖库。所有指标都是使用PyTorch autograd分析器得出的。torchprof仅在**PyTorch 1.8以下版本中**使用，<u>PyTorch 1.9及以上版本中使用Torch.profiler代替</u>。

```python
import torch
import torchvision
import torchprof

model = torchvision.models.alexnet(pretrained=False).cuda()
x = torch.rand([1, 3, 224, 224]).cuda()

with torchprof.Profile(model, use_cuda=True) as prof:
    model(x)

print(prof.display(show_events=False))         
```

> ```
> Module         | Self CPU total | CPU total | Self CUDA total | CUDA total | Number of Calls
> AlexNet        |                |           |                 |            |           
> ├── features   |                |           |                 |            |           
> │├── 0         | 5.551s         | 22.205s   | 5.551s          | 22.205s    | 1         
> ```

如果要查看每个层中发生的低级操作，可将`prof.display(show_events=False)`改为True

>```
>├── features                        |                |           |                 |            |                
>│├── 0                              |                |           |                 |            |                
>││├── aten::conv2d                  | 58.600us       | 213.896ms | 63.762us        | 213.894ms  | 1              
>... 
>```

可以通过在概要文件实例上调用raw()返回原始的Pytorch事件列表，需插入以下代码:

```python
trace, event_lists_dict = prof.raw()
print(trace[2])
print(event_lists_dict[trace[2].path][0])
```

如果要分析某一层，可以选择单独使用可选kwarg路径参数，忽略所有其他层的分析，代码如下：

```python
model = torchvision.models.alexnet(pretrained=False)
x = torch.rand([1, 3, 224, 224])
 
# Layer does not have to be a leaf layer
paths = [("AlexNet", "features", "3"), ("AlexNet", "classifier")]
 
with torchprof.Profile(model, paths=paths) as prof:
    model(x)
 
print(prof)
```

### rocprof介绍

rocprof为 ROCm 平台上类似 nvprof 的命令行工具，它本身是一个 shell 脚本，主要功能是设置进行性能分析所需的环境变量、动态库路径、输入输出文件名等信息，然后运行用户要分析的程序。实际抓取性能分析数据的功能由rocprofiler 和 roctracer 两个库实现，其中roctracer 库实现 API 时间线跟踪数据的抓取， rocprofiler 库实现 dcu 硬件计数器信息的抓取和以此为基础的各种性能指标 matrix 的计算。
rocprof 包括两类功能：

1. 提供时间线和热点函数追踪分析。分析应用程序每个线程执行的 hip 、 has 或者 kfd 函数调用和时间线，找到热点函数，并可以转换生成 json 文件，使用 chrome://tracing 在浏览器中可视化。
2. 采集每个 kernel 函数执行的 DCU 硬件计数器和不同的性能指标 matrix 。分析每个 kernel 函数执行所需资源，运行瓶颈等，可以辅助指导 kernel 函数的优化。

#### rocprof命令行参数简介

通用类
`-i <.txt|.xml file>`：输入文件名称，用于配置计数器或 Metrics 信息，支持 txt 和 xml 格式的输入文件。

`-o <output file>`：输出文件名称，输入文件为 CSV 文件，默认名称为 <input file base>.csv

`-d <data directory>`：性能分析工具产生的中间文件的存储目录，默认为 /tmp ，且默认情况下该目录在性能分析运行结束后自动删除

`-t <temporary directory> `：性能分析工具产生的中间文件的临时存储目录，默认为 /tmp ，且默认情况下该目录在性能分析运行结束后自动删除

时间分析类：

--hip-trace ：抓取 hip API 的运行信息，生成可以使用 chrome://tracing 可视化的 json 文件

--hsa-trace ：抓取 hsa API 的运行信息，生成可以使用 chrome://tracing 可视化的 json 文件

--sys-trace ：抓取 hsa /hip API 以及 GPU 活动的运行信息，生成可以使用 chrome://tracing 可视化的 json 文件

--kfd-trace ：抓取 KFD Thunk API 的运行信息，生成可以使用 chrome://tracing 可视化的json 文件

`--trace-start <on|off>` 设置是否从应用程序的起始部分开始trace

`--trace-period <dealy:length:rate>` 设置 trace 的初始延迟、周期性采样时间和频率。支持的时间格式为<number(m|s|ms|us)>

`--flush-rate <rate>`：设置 trace 结果输出的频率，支持的时间格式为<number(m|s|ms|us)>

计数器采集类
--verbose ：导出输出 matrix 所用到的所有基本计数器值
--list-basic ：打印当前支持的所有基本计数器名称、意义和数量
--list-derived ：打印当前支持的所有 matrix 名称、意义和计算公式

采集的计数器或matrix 需要通过输入文件指定。支持 txt 和 xml 格式的输入文件。

#### 使用实例

1. 编译程序：` hipcc simpleStream.cpp o simpleStream` 
2. 运行程序： `./simpleStream`
3. 使用 rocprof 运行程序：` rocprof hip trace ./simpleStream`
4. 运行完成后当前目录生成默认的以 results 开头的结果文件，包括：results.copy_stats.csv results.db results.hip_stats.csv results.json results.stats.csv
其中：
db文件是 sqlite 数据库文件，可用 sqlite 所支持的数据库查看工具打开，查看里面的全量数据。 csv 和 json 文件内容都是以数据库中存放的数据为基础汇总转换得出的结果。
csv文件分别是 hip/has API 以及 GPU 活动的 trace 结果，可以使用 Excel 查看。
json文件可以使用 chrome 浏览器进行可视化查看，详细查看各个 stream 中内存拷贝和 kernel 函数执行的过程。

