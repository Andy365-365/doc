# 3090 Ti 上的 FP8：消费级 GPU 上真正能用的路径

作者：[Wei Jie Chee](https://instavar.com/about#wei-jie-chee) / 发布时间：2026-05-10

原文：[FP8 on RTX 3090 Ti – What Actually Works on Consumer GPUs](https://instavar.com/research/model-deployment/fp8-on-24gb-gpus)（instavar.com，译文仅供参考，以原文为准）

**60 秒速览**
>  RTX 3090 Ti 上的 FP8 是真实可用的，但它主要是一个**省显存的存储技巧**，而不是一条 FP8 加速路径。
>  3090 Ti 是 Ampere GPU，计算能力 8.6。它可以把权重存在 `torch.float8_e4m3fn` 里，但原生 FP8 tensor core 计算不是你应该指望的路。消费级 GPU 上可靠的配方是：用 FP8 存大型 diffusion transformer 的权重，用 BF16 做计算。
>  模型如果差点就能塞进 24 GB，先用 diffusers 的逐层转换（layerwise casting）；还塞不下，就把 diffusion transformer 的 FP8 存储和文本编码器的 NF4 组合起来。

## 这份指南给谁看
 这份指南面向在单张 24 GB NVIDIA GPU 上跑图像或视频生成模型的人：
- RTX 3090

- RTX 3090 Ti

- A10

- RTX 4090

- L40 / L40S
 最常见的读者画像：模型差点就能跑起来。某个 README、issue 评论或 benchmark 说"用 FP8"，但那份配方很可能是在 4090、L40、H100 或更新的卡上写的。对 3090 Ti 来说，这个细节很重要。 问题不在于"PyTorch 有没有暴露 FP8 数据类型？"——有。真正的问题是： **哪些 FP8 路径在 Ampere 上真的有用，哪些默默假设了更新的硬件？**
## 简短的回答
 在 RTX 3090 Ti 上：
```python
pipe.transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)
```
 这是务实的路径。它把 transformer 权重存成 FP8，每层真正执行计算时再转回 BF16。 别指望它让生成变快。要指望的是：它把权重显存压得足够低，让更大的模型或更高分辨率能跑起来。
## SM 8.6 与 SM 8.9 到底差在哪
 SM 指 Streaming Multiprocessor 架构版本，在 CUDA 文档里通常叫 compute capability（计算能力）。 与本文相关的分界线：
| GPU 家族 | 代表型号 | 计算能力 | FP8 的实际含义 |
|---|---|---|---|
| Ampere 消费级 / 准专业级 | RTX 3090、RTX 3090 Ti、A10 | 8.6 | FP8 存储有用；计算应保持 BF16 或 FP16 |
| Ada | RTX 4090、L4、L40、RTX 6000 Ada | 8.9 | FP8 tensor core 路径开始有硬件意义 |
| Hopper | H100 | 9.0 | FP8 是一等公民级别的数据中心训练/推理路径 |
 NVIDIA 的 CUDA 调优指南把 Ampere 列为计算能力 8.0 和 8.6，Ada 为 8.9。NVIDIA 的 Ada 架构材料也明确提到第四代 Tensor Core 支持 FP8；而 Ampere 材料强调的是 TF32、BF16、FP16、INT8 和 INT4，不是 FP8。 同一块 GPU 上实测过的 INT4 例子，可参考我们的[MicroZoom 24 GB 推理与 LoRA 实验](https://instavar.com/research/model-deployment/microzoom-int4-lora-24gb-gpu-experiment-2026)。 这就是分界线。3090 Ti 对本地 AI 生产可以非常有用，但它站在原生 FP8 计算这条线的错误一侧。
## 为什么这事容易让人糊涂
 "FP8"这个词下面藏着三个不同的概念：
| 说法 | 含义 | 3090 Ti 上的结果 |
|---|---|---|
| 存在 FP8 数据类型 | PyTorch 能用 float8 dtype 表示张量 | 成立 |
| FP8 存储 | 权重以 FP8 保存，计算时上转 | 有用 |
| FP8 计算 | 矩阵乘走 FP8 tensor core kernel | 不是 Ampere 上可靠的路径 |
 大多数博客和仓库评论把这三者混为一谈。于是人们就会在 3090 Ti 上试 FP8 激活量化（activation quantization）的配方，然后纳闷为什么结果失败、回退、或产出质量差。 Ampere 的规则很简单： **用 FP8 存权重，用 BF16 或 FP16 做计算。**
## 在 RTX 3090 Ti 上验证过的配方
 最干净的路径是 diffusers 的逐层转换。它就是为这个场景设计的：模块权重用低显存的存储 dtype 保存，前向传播时逐层上转。
```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "your/model",
    torch_dtype=torch.bfloat16,
)

pipe.transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)

pipe.transformer.to("cuda")
pipe.vae.to("cuda")
```
 以 9B diffusion transformer 为例，心智模型是这样的：
| 组件 | BF16 显存 | FP8 存储显存 |
|---|---|---|
| 9B transformer 权重 | 约 18 GB | 约 9 GB |
 省下的显存就是收益。前向传播仍走 BF16 计算，所以别把它包装成加速技巧骗自己。
## 24 GB 扩散 pipeline 的组合模式
 对现代图像和视频模型来说，transformer 不是唯一的大头，文本编码器可以非常大。 24 GB 卡上最实用的显存余量组合是：
- Diffusion transformer：FP8 逐层转换

- 文本编码器：bitsandbytes 的 NF4

- VAE：BF16

- CPU offload：测过各组件实际大小之后再考虑
  FLUX 风格 pipeline 的示例：
```python
import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

text_encoder_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

text_encoder = TextEncoderClass.from_pretrained(
    repo_id,
    subfolder="text_encoder",
    quantization_config=text_encoder_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

pipe = PipelineClass.from_pretrained(
    repo_id,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16,
)

pipe.transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)
```
 9B transformer 加 8B 文本编码器，3090 Ti 上的大致显存预算：
| 组件 | 手段 | 大致显存 |
|---|---|---|
| Diffusion transformer | FP8 存储、BF16 计算 | 约 9 GB |
| 文本编码器 | NF4 双重量化 | 约 4 GB |
| VAE | BF16 | 约 0.2 GB |
| 激活与开销 | 取决于分辨率 | 约 2–3 GB |
| **总计** |  | **约 15–16 GB** |
 这样在 24 GB 卡上还剩真实的余量。
## 先别做的事

### 1. 别一上来就做 FP8 激活量化
 激活量化是计算路径的特性，不只是存储特性。硬件边界就是在这里咬人的。 在 3090 Ti 上，从逐层转换开始，计算 dtype 保持 BF16 或 FP16。
### 2. 别以为单张图的冒烟测试能证明批量运行没问题
 这个模式我们反复见过：
1. 第一张图过了。

2. 第二或第三张图 OOM。

3. 问题不是模型简单意义上"放不下"。

4. 问题是组件换入换出、分配器碎片，或重复 pass 的开销。
  如果运行依赖 CPU offload，用多图批量来验证。第一遍前向传播干净，不等于后面都干净。
### 3. 别以为预量化的 FP8 仓库就是一个 diffusers pipeline
 有些 FP8 模型仓库就是一个单独的 safetensors 文件。它可能有用的，但不等于一个完整的 diffusers 仓库——后者有 model_index.json、scheduler 配置、VAE 配置和组件子文件夹。 更稳妥的路径：
1. 加载完整的 BF16 或标准 diffusers 仓库。

2. 加载时应用量化。

3. 保存好能跑的加载脚本。
  单文件 FP8 checkpoint，留到你准备好处理格式转换和 state-dict 加载细节时再用。
## 技术路线对比（RTX 3090 Ti）

| 技术 | 显存效果 | 3090 Ti 上的速度效果 | 质量风险 | 适用场景 |
|---|---|---|---|---|
| BF16 基线 | 显存最高 | 参照基准 | 最低 | 模型本来就放得下 |
| FP8 逐层转换 | 权重显存约减 50% | 通常不会更快 | 低 | 模型几乎放得下 |
| torchao float8 weight-only | 显存目标类似 | 通常不会更快 | 低到中 | 需要 torchao 特定流程 |
| NF4 bitsandbytes | 削减更激进 | 可能更慢 | 中 | 文本编码器太大 |
| INT8 weight-only | 中等程度削减 | 通常不会更快 | 低到中 | FP8 路径别扭或不支持 |
| CPU offload | 降低 GPU 驻留峰值 | 更慢 | 低 | 组件只能一个个装下 |
 默认推荐顺序：
1. 放得下就先试 BF16。

2. 几乎放得下，就对 transformer 上 FP8 逐层转换。

3. 下一个瓶颈是文本编码器，就给那个组件用 NF4。

4. 组件总显存远超显存容量，再加 CPU offload，并且测多轮生成。

## 一份务实的决策树

### 模型在 BF16 下放得下
 默认不要量化。省下的工程时间和避免的意外质量漂移，比那点显存值钱。
### 只有 transformer 太大
 用 FP8 逐层转换：
```python
pipe.transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)
```
 这是 3090 Ti 上最好的第一步。
### 文本编码器太大
 给文本编码器用 `transformers.BitsAndBytesConfig`。注意它和 diffusers 的量化配置类是分开的两个东西：
```python
from transformers import BitsAndBytesConfig
```
 文本编码器通常比 diffusion transformer 更适合做 NF4：文本 embedding 路径上的小质量损失，往往比去噪 transformer 里的量化损伤更不明显。
### 两个组件各自放得下，但 pipeline 整体 OOM
 用组件级 offload，并用重复生成来验证。陷阱在于以为"每个组件都放得下"等于"整条跑稳定"——经常不是。
### README 说"用 FP8 提速"
 查一下那个 benchmark 用的什么 GPU。如果是 Ada、Hopper 或 Blackwell，别把速度结论搬到 Ampere 上。
## 两个 BitsAndBytesConfig 的坑
 有两个名字很像的配置类：
| 组件 | 正确的配置类 |
|---|---|
| Transformers 文本编码器 | `transformers.BitsAndBytesConfig` |
| Diffusers 模型组件 | `diffusers.BitsAndBytesConfig` |
 它们的参数名有重叠，但不通用。混用是快速拿到一团糊涂 loader 报错的捷径。 大多数 3090 Ti 扩散任务里，用下面这套组合就能绕开这个坑：
- 文本编码器用 `transformers.BitsAndBytesConfig`

- Diffusion transformer 用 `enable_layerwise_casting()`
  两个世界就分开了。
## 我们验证了什么
 本地验证环境：RTX 3090 Ti，24 GB 显存。 可沉淀的结论：
- FP8 逐层转换对大型 diffusion transformer 有效。

- 在 Ampere 上，它的价值是显存削减，不是 FP8 计算加速。

- 9B transformer 存成 FP8 后，在 24 GB pipeline 里变得实用。

- FP8 transformer 存储 + NF4 文本编码器量化的组合，能给 FLUX 风格的 pipeline 腾出足够余量。

- CPU offload 可能通过冒烟测试，但在组件预算太紧时，重复运行仍会失败。
  这部分才是值得带进生产的东西。
## 参考来源

- NVIDIA CUDA Ada 调优指南：[Ada 设备为计算能力 8.9](https://docs.nvidia.com/cuda/archive/13.0.2/ada-tuning-guide/index.html)

- NVIDIA Ampere 架构页：[Ampere Tensor Core 支持 TF32、BF16、INT8、INT4](https://www.nvidia.com/en-gb/data-center/ampere-architecture/)

- NVIDIA Ada 架构页：[Ada Tensor Core 新增 FP8 精度](https://www.nvidia.com/en-gb/technologies/ada-architecture/)

- NVIDIA TensorRT for RTX 支持矩阵：[计算能力 8.6 列出 FP8 不支持](https://docs.nvidia.com/deeplearning/tensorrt-rtx/1.3/getting-started/support-matrix.html)

- Hugging Face diffusers 显存文档：[torch.float8_e4m3fn 存储 + BF16 计算的逐层转换](https://huggingface.co/docs/diffusers/v0.33.1/en/optimization/memory)

- Hugging Face diffusers 模型文档：[模型级 enable_layerwise_casting](https://huggingface.co/docs/diffusers/api/models/overview)

- PyTorch torchao 推理文档：[float8 动态激活 + float8 权重工作流的要求](https://docs.pytorch.org/ao/stable/workflows/inference.html)

## Instavar 相关文章

- [24 GB GPU 上的声音克隆——2026 年真正能跑的方案](https://instavar.com/research/tts/voice-cloning-24gb-gpu-2026)

- [如何跑一场 AI 视频模型对比测试而不让它变成玄学](https://instavar.com/research/ai-video/how-to-run-an-ai-video-model-bakeoff-without-turning-it-into-vibes)

- [生产级 AI 视频 pipeline 到底需要什么](https://instavar.com/research/production-systems/what-a-production-grade-ai-video-pipeline-actually-needs)

- [HunyuanVideo 1.5——生产团队的升级清单](https://instavar.com/research/ai-video/hunyuanvideo-1-5-upgrade-checklist-for-production-teams)

## 一句诚实的话
 3090 Ti 上的 FP8 不是假的，只是比营销话术暗示的窄得多。 拿它来缩小权重，别指望 Ampere 表现得像 Ada 或 Hopper。把这个边界想清楚，FP8 存储就是让大型图像和视频生成模型在很多人已经拥有的硬件上变得实用的最简单的方式之一。
