# Arm-MNN-Qwen_1.8B的软硬件协同优化

## ✨&nbsp; 测评指南
模型转换已经完成，权重放置在`./data/final-qwen-chat`(合并版用于测试性能)和`./data/final-qwen-logits`(分离版用于测试精度，具体原因写在对应章节)。

性能，精确度的测评为严格按照官方示例逻辑而自主编写的脚本，在本节介绍使用方法，具体实现逻辑在各自对应章节详述。
- 精度
``` bash
cd /root/aicas2024
python eval-lm-mnn.py
```
运行后会保存文件到`results_mnn.json`中。

具体测评的实现逻辑在`lm-eval`中。
(为了确定测评的合理性，还针对hf模型按照相同逻辑编写了测评脚本`eval-lm-hf.py`作为对比实验)
- 性能 + 内存占用
```bash
python cpp-benchmark-monitor.py
```
调用cpp推理测试benchmark,将prefill/decode throughput保存在`throughput_mnn.json`中。

同时监测内存使用，将`max_rss`，`max_vms`保存在`memory_results.json`。
## 🤝&nbsp; MNN模型导出
模型转换文件主要基于llm-export项目。

由于精度测试需要求句子的loglihood，输出当前输入的所有字符的logits，而性能测试模型需要输出下一个字符的token_ids，具体是通过logits做argmax，因此导出了两个权重文件。同时为了确保所测精度和性能测试的权重一致，在模型导出时候，量化位数和模型结构保持严格一致。如需验证，可去转换文件中具体查阅。

导出逻辑:torch.nn.Module >> .onnx >> .mnn

保存在`./data/final-qwen-chat`(合并版用于测试性能)和`./data/final-qwen-logits`(分离版用于测试精度)。两者导出的逻辑和脚本一致。

- 性能权重导出
```bash
cd llm-export 
sh export-qwen.sh
```
主要技术：
1. torch_onnx_export导出为onnx，利用onnxslim进行模型自动剪枝。
2. onnx2mnn，利用[MNNConvert](https://mnn-docs.readthedocs.io/en/latest/tools/convert.html)将onnx模型转化为mnn模型，并进行4bits量化，以及内部的arm指令集优化。
3. 导出时embeddings导出为.bin文件保存在存储中，在加载时候减少内存占用。
4. 模型整个导出为一个.mnn和扩展的.mnn.weight文件，能够有效的减少内存占用。但是合并导出，会导致推理速度下降10%左右。

- 精度权重导出
```
cd llm-export 
python qwen_export-lm-eval.py
```
1. 分别导出各个对应部分

| hf   | mnn | 文件名    |
| ----- | ---- | ------- |
| transformer.wte   | self.embed_   | embedding.mnn   |
| transformer.h   | self.blocks_ | block_{i}.mnn   |
| transformer.ln_f   | self.final_layernorm_  | 合并在block_24    |
| model.lm_head   | self.lm_   | lm.mnn    |

2. 相对于性能权重，在最后一个block，不进行取最后一个字符的操作，而是保存所有的logits。并且在lm部分不进行argmax，而是保留原始的logits用于在lm-eval中对数似然。

## 🚀&nbsp;benchmark与monitor测试性能
- benchmark

主要利用[mnn-llm](https://github.com/wangzhaode/mnn-llm)作为基础来进行推理，调用`mnn-llm/demo/cli_demo.cpp`文件，加载上prompt.txt作为输入，来进行测试。生成第一个字符作为prefill的时间，然后进行decode，直到生成足够50个字符进行强制暂停，作为decode时间。cmake构建的文件放置在build中。
```bash
cd mnn-llm/build
./cli_demo ../../data/final-qwen-chat/llm.mnn ../../prompt.txt
#如果您使用vscode进行连接，可以直接使用我配置的launch.json.需要注意的是vscode连接会占用内存，外加我安装了灵码的插件会影响速度。
```
具体的实现逻辑在`mnn-llm/src/llm.cpp`中。是在cli_demo中首先加载llm类，然后调用response方法。
主要技术：

1. 多线程：经过实验发现，在倚天710上设置8线程运行速度最高，高或者低都会下降性能。
2. mnn的module推理。这部分涉及mnn的具体知识，概括为预推理机制和算子优化。

- monitor + benchmark

使用python的subprocess技术来运行cli_demo可执行文件，测试速度，并设置输入和输出，之后通过Thread库，设置另一个线程专门用于检测内存占用情况，将结果写入到memory_results.json中。
```bash
python cpp-benchmark-monitor.py
```

### <span style="color:red;">!</span> 关于kv-cache管理

在经过上述优化方案后，我发现prefill和decode能够达到320/11 tokens/s，内存占用能够下降到3G以内。接下来进行性能瓶颈分析，发现主要耗时在kv-cache的读写，采用了渐进遗忘kv-cache的策略。
即在decode过程中，生成新的kv-cache的同时，清除掉一些过于长程的kv-cache。具体实现在src/llm.cpp中，如果改动，请利用script/rebuild.sh进行重新编译。

```cpp
// 采用渐进遗忘的推理策略,1渐进遗忘，2全部遗忘，3不遗忘
int control = 1;
if (control == 1) 
{...}
else if (control == 2) 
{...}
else if (control == 3)
{...}
```
经过我的实际测试，如果采用我设置的渐进遗忘(control=1)，那么吞吐能够提升到320/300，生成结果仍然合理，没用任何乱码，并且渐进遗忘也保证了一开始能够prefill看到全文信息，具备合理性。

如果不进行任何的kv-cache管理(control=2)，吞吐为320/11,生成结果显式的表达出来。

如果直接将kv-cache置空，那么将会极大程度的减少内存占用并且提升吞吐到300/60,但是这样decode出来的结果实际上全是,,,,,,。并不具备商用价值，我看到排行榜单上许多队伍是这样的成绩，可能只是巧合，也有可能采用了这样的方式。如果采用这样方法，个人认为并不合理，只是实现了decode速度，并没有实际应用的价值。

我在这里对另一个框架llama.cpp进行了调研，发现有两个参数会影响kv-cache的管理,例如：
```python
from llama_cpp import Llama
llm = Llama(model_path="zephyr-7b-beta.Q4_0.gguf", n_ctx=512, n_batch=126)
```
n_ctx：模型的最大上下文大小，n_ctx是输入提示（prompt）中的token数量与模型能够生成的最大token数量之和。上下文大小较小的模型生成文本的速度会比上下文大小较大的模型快得多。

n_batch：用于设置在生成文本时一起批处理的最大提示token数量，默认值也是512个token。降低n_batch有助于在多线程CPU上加快文本生成速度。如果输入提示（prompt）的token数量超过了n_batch的大小，它会被分割成多个批次进行处理。提示不会被截断，而是分成多个部分，每个部分包含不超过n_batch个token。这样，模型可以同时处理多个这样的部分，从而提高整体的生成效率。

实际上选手们只需要把这两个参数调的足够小就能拿到300/50的成绩，但生成的文字结果并不正确或合乎逻辑。而我设计的基于mnn框架的渐进遗忘策略类似于llama.cpp的n_ctx参数设置，但是动态的kv-cache管理以及我根据实验多次测试出的参数，确保了生成的decode结果合理。个人建议是为了公平起见，应当保证选手们decode出来的结果不是乱码，并且合乎逻辑，而不是只看token_ids是否添加进入张量。

## ✅&nbsp;lm-eval测试精度
[lm-eval自定义model文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)

编写了测试逻辑，主要是对于多项选择题piqa，对各个选项进行loglikelihood，返回logits得分较高的选项作为答案。

编写了测试类放在`/root/aicas2024/lm-evaluation-harness/lm_eval/models/mnn.py`（通过装饰器与测评脚本互动）。调用测试脚本直接得到结果。
```bash
python eval-lm-mnn.py
```

## 📫&nbsp; 安装依赖
比赛提供的倚天710服务器所有配置环节已经完成

## ❤️&nbsp; 相关开源项目及人员

1. [mnn-llm](https://github.com/wangzhaode/mnn-llm)
2. [MNN](https://github.com/alibaba/MNN)
3. [llm-export](https://github.com/wangzhaode/llm-export)
4. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

特别鸣谢mnn-llm的开发者wangzhaod提供的思路，以及我的导师zhuhuming老师给予的指导和鼓励，以及lm-eval开发者Hailey Schoelkopf在我编写测评脚本时提供的帮助。

## 📘&nbsp; 引用

待完善
