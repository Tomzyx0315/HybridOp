# hd-4 实验机

2026-11-17 准备完成并启动正式训练；2026-11-18 按用户要求停止队列，检查 ZO 梯度。
当前停止状态与诊断结果见本文末尾。以下路径均在远端机器。

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

下载、安装和验收均在独立远端 tmux 会话中执行，准备任务已结束。
后续训练也应在 tmux 中启动。例如进入一个新的会话后加载环境，再选择启动脚本：

```bash
tmux new -s hybridop-train -c /data/zhangyuxuan/HybridOp
source /data_fs/zhangyuxuan/HybridOp/setup/env.sh
# 在这里手动选择 full_fo / truncated / 某种 hybrid 启动脚本。
```

Mac 合盖或 SSH 断开不会终止 tmux 中的任务；重新连接后用
`tmux attach -t hybridop-train` 返回会话。助手需要恢复连接后才能继续查看状态。

## 2026-11-18 停训与梯度诊断

正式实验目录：`/data_fs/zhangyuxuan/HybridOp/experiments/formal_20261117_215621`。
完整 FO、truncated 均完成 19,523 步，最终 34,560 行验证 loss 分别为
2.43435、2.50287。Hybrid none 在最后记录约 9,890 步时按用户要求停止；最近
完整 checkpoint 为 8,000 步。整个队列一起停止，separate/joint_recompute 未启动。
`status_before_stop.json`、`stop_request.json` 和 `status.json` 保存停止记录。

诊断目录：`/data_fs/zhangyuxuan/HybridOp/diagnostics/gradient_20261118_103632`。
使用 Hybrid none 的 8,000 步 checkpoint、下一批真实数据、8 卡 × 8 次累积，
完整重放 262,144 tokens；eager 和 compiled 两次检查均成功。仅保存 JSON 指标，
优化器反事实更新只在内存中进行，没有覆盖 checkpoint 或恢复正式训练。

编译路径下，8 卡平均后的共享参数 suffix FO 范数为 0.5294，精确早期参数 VJP
为 0.01558，BF16 ZO 为 65.084。用精确方向导数构造同样的高斯随机方向估计，
范数仍为 65.302；FP32 差分为 65.335。BF16 ZO 与精确早期 VJP 的 cosine
约 0.000311。被扰动的共享参数共 102,777,856 维，每次更新仅 64 个独立方向
（每卡、每个 microbatch 一个方向，分别作用于不同数据）。由精确 microbatch
梯度计算的高斯估计器 RMS 范数预测为 62.526，与实测一致到采样波动量级。

这些结果支持高维随机方向估计的方差是当前主要问题；未发现累积/跨卡平均漏除
系数。BF16 在默认 epsilon 下的差分误差相对次要，单独改 FP32 不能消除该噪声。
eager 优化器对照中，恢复同一历史状态后，精确早期参数梯度与实际 ZO 的共享参数
更新范数约为 1.221 / 1.224，但方向 cosine 约 0.739。清零历史矩以隔离当步梯度
后，cosine 降至约 0.000692，显示 MuonH 控制模长并不保证估计方向可靠。
该诊断限定于这个 checkpoint 和一个 batch，不能当作 lambda 收敛实验。

原始结果位于上述诊断目录下的 `eager/summary.json`、`compiled/summary.json`。

### 完整 FO checkpoint 对照

2026-11-18 追加检查，目录：
`/data_fs/zhangyuxuan/HybridOp/diagnostics/fo_checkpoint_20261118_105446`。
读取 full_fo 的 8,000 和 16,000 步权重，统一使用 8,000 步后的一个完整 batch，
并固定 direction_step=8000、epsilon=1e-3、8 卡 × 8 次累积、编译模式。
已核对 FO/Hybrid 的 8,000 步每卡数据游标相同；两次 FO 检查实际输入的逐 rank
SHA256 完全一致。两次诊断退出码均为 0，没有运行 optimizer.step 或保存权重。

下表为 8 卡平均后的首轮共享参数梯度；“精确导数 ZO”表示使用精确方向导数、
但仍乘回随机方向来构造估计，不包含有限差分误差。

| checkpoint | 精确首轮梯度范数 | BF16 ZO 范数 | 精确导数 ZO 范数 | ZO 与精确首轮梯度 cosine |
| --- | ---: | ---: | ---: | ---: |
| Hybrid none / 8,000 | 0.01558 | 65.08 | 65.30 | 0.000311 |
| 完整 FO / 8,000 | 0.07857 | 755.76 | 762.14 | 0.000292 |
| 完整 FO / 16,000 | 0.08969 | 859.68 | 861.02 | 0.000176 |

FO 的 FP32 ZO 范数分别为 758.93、856.82；高斯估计器预测 RMS 范数分别为
666.75、823.47。各 microbatch 精确首轮梯度平方范数的均值，相对于全局均值梯度
平方范数的比值，Hybrid 为 10.03，FO 两个 checkpoint 为 44.85、52.49。
完整 FO 训练并未在这两个 checkpoint 上消除当前估计器的高方差，且首轮梯度
与 microbatch 差异更大，使绝对噪声范数更大。这里仅比较一个固定 batch 的梯度，
尚未测试从 FO checkpoint 切换 ZO 后的长期收敛。

完整数据见 `comparison.json`、`full_fo_8000/summary.json`、
`full_fo_16000/summary.json`。诊断结束后所有 GPU 已释放，正式队列仍停止。

## 2026-11-18 范数匹配：短跑与从零训练

短跑目录：`/data_fs/zhangyuxuan/HybridOp/experiments/norm_match_20261118_110722`。
六组均恢复完整 FO 的 8,000 步权重、分片优化器状态和数据游标，各跑 200 步，
保留原 19,523 步学习率日程。所有分支的初始验证 loss 和首批逐 rank SHA256
一致；每组在 0/50/100/200 步评估固定 4,992 行。退出码为 0，权重单独保存，
没有覆盖原正式实验。初始验证 loss 为 2.64948813。

| 分支 | 200 步后验证 loss |
| --- | ---: |
| 完整 FO | 2.64685859 |
| truncated | 2.67375932 |
| ZO/FO 范数比 1 | 2.64864058 |
| ZO/FO 范数比 0.3 | 2.65357118 |
| ZO/FO 范数比 0.1 | 2.65615631 |
| 原始 Hybrid，lambda=1 | 2.69428793 |

范数匹配在这个短跑中优于原始 Hybrid 和 truncated，但没有优于完整 FO；比值 1
在第 50 步一度低于完整 FO，200 步时优势未保持。单个 seed、200 步和 FO checkpoint
切换不能支持从零预训练或长期收敛的结论。原始记录在 `short_runs/*/metrics.jsonl`，
汇总在 `short_runs/summary.json`，核对记录为 `verification.json`，曲线为 `short_runs.png`。

本仓库也归档了本次短跑的[结果汇总](results/20261118_norm_match/summary.json)、
[梯度诊断](results/20261118_norm_match/gradient_diagnostic.json)和
[验证曲线](results/20261118_norm_match/short_runs.png)。同目录下的 `manifest.json`
记录运行配置，各分支子目录保存 `metrics.jsonl` 和首批数据 SHA256；不包含权重或数据集。

单批诊断在 `diagnostic/summary.json`。共享参数 FO 范数 0.18557、首轮精确梯度
0.07858、ZO 范数 755.55；等范数匹配的有效 lambda 为 2.456e-4。
若以无偏高斯估计模型下的梯度均方误差为目标，利用各 microbatch 精确梯度计算
的 ZO RMS 666.75，最优固定 lambda 约为 1.39e-8，对应 ZO/FO 范数比约 5e-5。
这是该 batch 的梯度估计诊断，不是 MuonH 训练 loss 的最优超参；范数匹配本身也
不保持原估计器的无偏性。64 个方向来自 8 卡 × 8 次累积，各 microbatch 一个方向，
不是同一批数据探测 64 次；每个 microbatch 仍只有两次首轮循环探测前向。

用户决定改为原日程的从零训练。新实验目录：
`/data_fs/zhangyuxuan/HybridOp/experiments/norm_pretrain_20261118_112549`。
tmux 为 `hybridop-norm-pretrain`，依次运行 `norm_match_1`、`norm_match_0.3`、
`norm_match_0.1`，每组 19,523 步、每 250 步验证、每 2,000 步保存。
首组已启动，初始验证 loss 10.373580708717688 与原 FO/truncated 完全一致，
首步训练 loss 也与 FO 一致；启动验收时已检查到 100 步，数值有限，日志中缩放后
ZO 范数满足目标比例。这是启动时记录，后续进度以服务器上的 `status.json` 为准。
初始化首步边界梯度与 ZO 均为零，缩放系数为零。

保留 seed=42、模型/数据/优化器/原学习率日程、epsilon=1e-3、每卡每 microbatch
一个方向、prelude_mode=none。FO/truncated 复用原从零训练结果，未重新运行。
`checks/baseline_comparison.json` 记录配置及核心文件一致性；43 项 CPU/配置/续训
检查和两卡混合梯度/更新检查通过，日志在 `checks/`。源码快照及哈希分别在
`source/` 和 `manifest.json`，队列状态在 `status.json`，各分支日志和 checkpoint
在 `<分支>/console.log`、`<分支>/run/`。入口位置记录于
`/data_fs/zhangyuxuan/HybridOp/setup/latest_norm_pretrain_experiment.txt`。

正式入口为 `hybrid/scripts/hybrid_norm_match.sh`，支持 checkpoint 续训；新增比例
写入 `user_config.zo_norm_ratio`。此实现额外保留 ZO 梯度缓冲区并先平均共享 FO/ZO，
尚未优化额外通信，不能把短跑耗时当作最优实现的吞吐比较。所有权重、日志和缓存
仍在 `/data_fs/zhangyuxuan/HybridOp` 下；原正式实验队列保持停止。

### 2026-11-18 18:37（北京时间）进度快照

范数比 1 于北京时间 17:28 完成全部 19,523 步，退出码为 0，最终权重、metadata
及 8 卡优化器/数据游标文件均已核对存在。最终评估统一使用 34,560 行：

| 方法 | 最终验证 loss |
| --- | ---: |
| 完整 FO | 2.43435021 |
| truncated | 2.50286627 |
| 范数比 1 | 2.65894234 |

范数比 1 相对 truncated 高 0.15608；从 1,000 步起的各次相同步数验证也持续落后。
在当前配置和 seed 下，等范数缩放控制了梯度大小，但没有带来训练收益。

范数比 0.3 正在运行，快照记录到 3,750 步，当时最近完整验证为 3,500 步；随后
检查进度到 3,790 步。范数比 0.1 仍排队。3,500 步统一使用 4,992 行评估：

| 方法 | 3,500 步验证 loss |
| --- | ---: |
| 完整 FO | 2.80384908 |
| truncated | 2.87811132 |
| 范数比 0.3 | 2.93987061 |
| 范数比 1 | 2.98443239 |

0.3 比 1 有改善，但尚未优于 truncated；不据此推断它或 0.1 的最终结果。
两组日志没有非有限数值，记录的缩放后 ZO/FO 范数比均满足设置。范数比 1 最后
100 条训练记录的原始 ZO 范数约 138–225，有效 lambda 约 7.38e-4–1.24e-3；
0.3 最近 100 条记录约 140–262、2.02e-4–4.41e-4。当前 8 卡利用率约 98–99%，
每卡进程占用约 8.2 GiB，每步约 0.98 秒。检查未改变训练配置或队列。

本地归档：[完整快照与验证序列](results/20261118_norm_pretrain_1836/summary.json)、
[对齐步数的验证曲线](results/20261118_norm_pretrain_1836/validation.png)。曲线仅使用
4,992 行评估的点，并从 1,000 步开始显示；最终 34,560 行结果单独列于上表。
服务器同份报告位于本轮实验的 `reports/20261118_1836/`。

### 2026-11-18 20:30（北京时间）进度快照

范数比 0.3 已完成 10,000 步验证（总日程 19,523 步），范数比 1 已完成，
范数比 0.1 尚未开始。以下均为 10,000 步、同一组 4,992 行验证数据：

| 方法 | 10,000 步验证 loss |
| --- | ---: |
| 完整 FO | 2.60624049 |
| truncated | 2.67866092 |
| 范数比 0.3 | 2.72827868 |
| 范数比 1 | 2.80797151 |

0.3 相对 truncated 的差距为 +0.04962，较 3,500 步时的 +0.06176 略小，但仍没有
优于截断。相对范数比 1 的改善已持续到 10,000 步，最终结果仍需等该组训练完成。
未发现 NaN/Inf 或范数匹配错误，8 卡正常运行，每步约 0.97 秒。按已运行时间
线性估计，0.3 组约于当日 23:23 完成，随后开始 0.1；该时间仅是当前速度下的估计。
检查未更改参数、训练进程或队列。

本地归档：[结果快照与验证序列](results/20261118_norm_pretrain_2030/summary.json)、
[更新后的验证曲线](results/20261118_norm_pretrain_2030/validation.png)。服务器报告位于
本轮实验的 `reports/20261118_2030/`。

### 2026-11-18 22:50（北京时间）进度快照

范数比 0.3 快照记录到 17,850 / 19,523 步，随后确认已继续到 17,880 步；范数比 1
已完成，0.1 仍在队列中。最近完整验证为 17,750 步，统一使用 4,992 行：

| 方法 | 17,750 步验证 loss |
| --- | ---: |
| 完整 FO | 2.45711262 |
| truncated | 2.52598071 |
| 范数比 0.3 | 2.56809512 |
| 范数比 1 | 2.67629127 |

0.3 相对 truncated 的差距为 +0.04211，比 10,000 步时的 +0.04962 继续缩小，
但至今仍未超过截断。按当前速度估计约 23:20 完成，之后队列会启动 0.1。
0.3 组日志无 NaN/Inf，范数匹配比例检查全部通过。最近 100 条训练记录的原始
ZO 范数约 393–670，缩放后约 0.0488–0.0576，有效 lambda 约 7.79e-5–1.33e-4，
每步中位耗时约 0.97 秒。复查时 8 卡利用率为 97–100%，未改变训练配置或进程。

本地归档：[结果快照与验证序列](results/20261118_norm_pretrain_2250/summary.json)、
[更新后的验证曲线](results/20261118_norm_pretrain_2250/validation.png)。服务器报告位于
本轮实验的 `reports/20261118_2250/`。
