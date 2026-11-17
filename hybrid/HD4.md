# hd-4 实验机

2026-11-17 准备完成，正式训练尚未启动。以下路径均在远端机器。

| 用途 | 路径 |
| --- | --- |
| 代码 | `/data/zhangyuxuan/HybridOp` |
| 所有新增大文件 | `/data_fs/zhangyuxuan/HybridOp` |
| FineWeb-Edu | `/data_fs/zhangyuxuan/HybridOp/datasets/FineWebEdu/prepacked_T2048_llama` |
| Python 环境 | `/data_fs/zhangyuxuan/HybridOp/envs/train` |
| 缓存 / 临时文件 | `/data_fs/zhangyuxuan/HybridOp/cache`、`tmp` |
| checkpoint / 日志 | `/data_fs/zhangyuxuan/HybridOp/checkpoints`、`logs` |
| 环境入口、准备脚本和依赖快照 | `/data_fs/zhangyuxuan/HybridOp/setup` |

## 使用环境

```bash
ssh hd-4
cd /data/zhangyuxuan/HybridOp
source /data_fs/zhangyuxuan/HybridOp/setup/env.sh
```

环境入口会设置 `PYTHON_BIN`、`NANOCHAT_BASE_DIR`、HF/编译缓存、临时目录和日志目录。
多卡编译时，训练入口把 Inductor/Triton 缓存分别放到所配置目录的 `rank0`、`rank1`
等子目录，避免各 rank 在共享文件系统上同时生成同一缓存文件。
`prepacked_T2048_llama` 和 `hybrid_runs` 分别链接到上面的数据和 checkpoint 目录，
因此原启动脚本的默认路径也会落在 `/data_fs`。

采用已有的 Python 3.12.3，通过独立 venv 复用系统 CUDA 库；在隔离层安装
PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128、Triton 3.5.1 等必要依赖。
原 `awm` 的 Python 3.10 / PyTorch 2.6 和系统 PyTorch 2.8 均未修改。
现有 PyTorch 2.8 缺少本实现使用的 `matmul.fp32_precision` 接口；低版本需要兼容适配。
完整版本见 `setup/environment.freeze.txt`。当前 attention 使用 PyTorch SDPA。

Tokenizer 已缓存，环境默认离线读取，训练不依赖 Hugging Face 连通性。
准备脚本使用已有网络加速，并支持 `hf-mirror.com` 回退；不会保存 SSH 凭据。

## 数据子集与兼容性

来源：[预打包 FineWeb-Edu / Llama 2](https://huggingface.co/datasets/KristianS7/prepacked-fineweb-edu-llama2-32K-T2048)。

- 固定 revision：`be2b9623f0cefa5146392bd48a71d35ea438d94d`。
- 训练：`train-00000` 至 `train-00169`，170 片，2,550,000 条序列。
- 每条提供 2048 个预测目标，共 **5,222,400,000 训练 tokens**，覆盖当前 5,117,837,312 tokens 的训练计划。
- 验证：全部 3 片、34,608 条序列。
- Parquet 总大小：9,947,722,356 bytes，约 9.95 GB。
- 每片均校验来源 SHA256、文件大小、2049 长度、BOS=1 和 token ID 范围。

原数据元数据声明 vocab_size=32008，但选取的所有序列均只使用 `<32000` 的基础词表 ID。
当前代码使用相同 Llama 2 基础词表并额外添加一个 token，因此运行时 vocab_size=32001。
Parquet 内容没有修改；子集元数据记录实际行数、运行时词表大小和原词表大小，原始元数据
保留在 `source_metadata/`。完整文件清单和校验结果见 `manifest.json`，完成标记为 `READY.json`。

Tokenizer 为 `NousResearch/Llama-2-7b-hf` 的 tokenizer.json，revision
`8efe6c9b93655b934e27bd9981e3ec13e55aee9d`；没有下载该模型的权重。

## 已完成的验收

- 57 项 CPU 数值/配置/续训测试，以及 12 项 CUDA 检查。
- 四种 Hybrid 的两卡梯度平均与优化器更新检查，各 rank 参数一致。
- 完整 FO、truncated、四种 Hybrid 各完成 4 步的 8 卡小模型 smoke，包括保存 checkpoint。
- 离线 tokenizer、真实数据 loader 检查。
- 六种方案的实际 218,786,816 参数模型单批检查：B=2、T=2048，前向、反向及一次更新均有限。

主要结果见 `logs/gpu_tests.log`、`logs/distributed.log`、`logs/final_checks.log` 和
`setup/check_219m.json`。小模型 smoke 输出在 `checkpoints/smoke_hd4_*`，不是正式实验结果。

219M 单批自检使用 eager、单卡未分片优化器，并将测试实例的 injection 状态部分设为
0.1I，以覆盖非零早期梯度；测试模型没有保存。下表为 PyTorch peak allocated，
不能当作正式 8 卡、编译运行的稳态显存或吞吐基准：

| 方案 | 峰值 GiB |
| --- | ---: |
| 完整 FO | 8.01 |
| truncated | 6.14 |
| Hybrid none | 6.14 |
| Hybrid separate | 6.14 |
| Hybrid joint | 6.14 |
| Hybrid joint_recompute | 5.54 |

GPU 检查修复了早期 no_grad 的 autocast 缓存使末轮共享参数丢失 FO 梯度的问题。
另有一个实验精度因素：BF16 两点差分会放大编译舍入差异，不能要求编译/eager 的
ZO 梯度逐元素接近。FP32 探测已与精确局部 VJP 对照通过；默认仍保留 BF16、
epsilon=1e-3、rho=1、lambda=1，探测精度和扰动幅度可在正式实验前决定。

## tmux 与停止状态

下载、安装和验收均在独立远端 tmux 会话中执行，准备任务已结束，没有启动正式训练。
后续训练也应在 tmux 中启动。例如进入一个新的会话后加载环境，再选择启动脚本：

```bash
tmux new -s hybridop-train -c /data/zhangyuxuan/HybridOp
source /data_fs/zhangyuxuan/HybridOp/setup/env.sh
# 在这里手动选择 full_fo / truncated / 某种 hybrid 启动脚本。
```

Mac 合盖或 SSH 断开不会终止 tmux 中的任务；重新连接后用
`tmux attach -t hybridop-train` 返回会话。助手需要恢复连接后才能继续查看状态。
