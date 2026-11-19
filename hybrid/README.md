# Hybrid loop training

本目录集中存放三种训练模式、Hybrid 的四种 prelude 处理、配置、启动脚本和验收测试。复用原仓库的
`GPT`、预打包数据管线、MuonH/AdamW 和 checkpoint 格式，不修改 `nanochat/`、
`scripts/` 或原有 `shells/`。

## 固定实验配置

配置文件：[`configs/loop2_219m.json`](configs/loop2_219m.json)。

| 项目 | 配置 |
| --- | --- |
| 模型 | 2 层 prelude + 8 层共享 block × 2 + 2 层 coda |
| 宽度 / attention | 1024；8 个 Q heads、8 个 KV heads；head dim 128 |
| 总参数量 | 218,786,816，含独立 embedding / lm_head |
| FFN | ReLU²，intermediate size 4096 |
| 循环输入 | `inject_init_prelude`，`W=[I,0]` 初始化 |
| 其他架构 | 保留 QK norm、learned RMSNorm、RoPE、softcap=15、原初始化和残差系数 1 |
| 数据 / tokenizer | 原仓库预打包 FineWeb-Edu / Llama 2 tokenizer，vocab_size=32001 |
| context / 全局 batch | 2048 / 262,144 tokens |
| 8 GPU 默认执行 | 每卡 2 sequences × 累积 8 次 |
| 精度 | CUDA BF16 autocast；沿用 embedding BF16；不启用 FP8 |
| 训练量 | 原代码 T/P=20：19,523 steps，实际 5,117,837,312 tokens |
| seed | 初始化 42；ZO 独立 seed 12345 |
| 学习率日程 | 无 warmup；全程线性衰减至初始 LR 的 10% |
| 验证 | 每 250 steps；中途 4992 rows，完整训练结束 34560 rows |

矩阵使用 MuonH（momentum=0.95、Polar Express 迭代 5 次、固定初始 Frobenius
范数），embedding/head/norm 使用 AdamW（betas=(0.8,0.95)、eps=1e-10、WD=0）。
不新增梯度裁剪。基本计算图编译复用原模型的 `compile_blocks()`，三种模式一致，
可用 `--no-compile` 关闭。

| 参数组 | 原始基础 LR | 默认实验实际初始 LR |
| --- | ---: | ---: |
| 矩阵 | 0.014 | 0.0070745097 |
| embedding | 0.3 | 0.2371708245 |
| lm_head | 0.004 | 0.0031622777 |
| norm | 0.005 | 0.0039528471 |

`config.py` 保留 `scripts/base_train.py` 的 batch/data/width 缩放公式。
其中原代码虽然构造 r=4 的参考模型，但计算其有效参数时显式传入**训练模型的
num_recur**；这里照抄这一实际行为，默认参考训练量为 606,380,800 tokens。
norm 的 LR 与原实现一样只做 width 缩放。启动时记录解析后的全部数值，避免把
0.014 误当作实际使用的 LR。

## 三种模式与局部梯度

令 `e=P(x)`，`A_theta(e)` 为早期循环的输出，`B_theta(e,h)` 为末尾 FO 循环
及后续预测。默认早期 1 轮、末尾 1 轮；`--fo-loops k` 支持末尾连续 k 轮。
`theta` 包括共享 recurrent block、共享 injection、共享 `norm_recur`。

| 模式 | 梯度处理 |
| --- | --- |
| `full_fo` | 原模型完整反传全部循环 |
| `truncated` | 早期循环在 `no_grad` 下执行，只反传末尾连续 FO 循环 |
| `hybrid` | truncated 的梯度 + 局部 ZO 补偿，之后统一执行一次优化器更新 |

默认 Hybrid（`--prelude-mode none`）固定 `e` 和边界梯度 `v = dL/dh`，
每个方向 `z ~ N(0,I)` 使用：

```text
c = <v, A_(theta + epsilon*z)(e) - A_(theta - epsilon*z)(e)> / (2*epsilon)
g_shared = g_suffix + lambda * average_directions(c*z)
```

不重新计算 perturbed coda/loss；这估计的是局部标量目标
`<stop_gradient(v), A_theta(stop_gradient(e))>` 的梯度。多轮早期循环作为一个
复合函数探测，每次出现的共享参数使用同一个扰动。有限 epsilon 存在差分/平滑误差。

三组均保留相同架构与前向值。初始化时 `W_h=0` 导致边界梯度为零是预期行为，
应在离开初始化点后检查估计质量。

## 四种 prelude 处理

`--prelude-mode` 只用于 Hybrid。四种均保留末尾 injection 到 prelude 的直接 FO
梯度，区别在于是否、以及怎样估计经过早期循环的那条输入梯度。这里 prelude
包含输入 embedding 和 `norm_emb`。设 `a` 为末尾 FO 对 `e` 的直接梯度，
`b = (dA_theta/de)^T v` 为缺失的输入梯度：

```text
g_prelude = (dP/dphi)^T [a + lambda * b_estimate]
```

| prelude_mode | 早期输入梯度 | prelude 图的处理 | 每方向的早期循环探测前向数 |
| --- | --- | --- | ---: |
| `none`（默认） | 不补回，即 b_estimate=0 | 随末尾 FO 反传并释放 | 2 |
| `separate` | 独立输入 ZO；原参数 ZO 不变 | 保留图，等输入估计完成后反传一次 | 4 |
| `joint` | 与参数 ZO 共用一对联合探测 | 保留图，等输入估计完成后反传一次 | 2 |
| `joint_recompute` | 与 `joint` 相同 | 首次 prelude 无图，探测后重算并反传一次 | 2 |

`separate` 在原参数正负探测之外，固定参数，额外计算
`A_theta(e ± epsilon_e*u)`，用其输出差与 `v` 的内积估计 `b`；`u ~ N(0,I)`。
`e` 的扰动同时进入早期循环的初始状态和所有早期 injection，末尾 FO 不重跑。

联合探测对参数、输入使用独立的高斯方向 `z,u`：

```text
d = <v, A_(theta + epsilon_theta*z)(e + epsilon_e*u)
        - A_(theta - epsilon_theta*z)(e - epsilon_e*u)>
g_shared += lambda * average_directions[d / (2*epsilon_theta) * z]
b_estimate = average_directions[d / (2*epsilon_e) * u]
```

使用 `--zo-rho` 显式控制 `rho = epsilon_e / epsilon_theta`，默认 1，即
`epsilon_e = rho * epsilon_theta`。增大 rho 会增大输入对参数估计的交叉噪声，
减小参数对输入估计的交叉噪声；减小 rho 则相反。此默认值尚未经过调参。
两个扰动尺度不能共用同一个分母。联合估计会带来参数与
输入方向的交叉噪声，因此它与 `separate` 的单次估计不相同；输入、参数方向独立时，
交叉项在一阶近似下期望为零。有限扰动和少量方向仍有误差与方差。

**`joint` 与 `joint_recompute` 使用相同种子、方向和估计公式，数学上得到相同梯度。**
区别是图的生存期和额外一次 prelude 前向。当前模型没有 dropout，重算期间也不更新
参数；若以后增加随机层，需要增加 RNG 重放。浮点/编译差异需在实验机验收，不能预先
保证 GPU 上逐位相同。`lambda=0` 时四种均直接走原 truncated 路径，不做探测或重算。

三种补回方案都对 `a + lambda*b_estimate` 一起反传 prelude，只做一次 prelude
backward。梯度累积的 `1/accumulation_steps` 已包含在 `a,v` 中，不重复除；
lambda 对参数补偿和输入补偿各乘一次。lambda=1 也不消除 ZO 的估计误差。

## ZO 超参数及文献依据

- `--zo-epsilon 0.001`：坐标尺度，即 `theta ± epsilon*z`，`z` 是未做单位范数
  归一化的标准高斯向量；不是相对权重范数的扰动。
- `--zo-rho 1.0`：输入与参数扰动的比例，输入尺度为 `epsilon_e = rho * zo_epsilon`。
  只用于三种补回 prelude 的方案；默认输入尺度为 0.001，是初始实验选择，
  并非 MeZO 文献验证过的激活扰动设置。rho 与 lambda 可独立调整。
- `--zo-input-epsilon`：保留绝对输入尺度的便捷入口，自动换算成 rho；不能和
  `--zo-rho` 同时指定。配置、checkpoint 和日志记录 rho，并记录解析后的输入尺度。
- `--zo-lambda 1.0`：按约定默认 1，可调整。设为 0 时跳过探测，严格退化成 truncated。
- `--zo-directions 1`：每卡、每个 microbatch 1 个方向。正负探测、所有共享参数和
  梯度重建使用同一方向。不同卡/microbatch 独立；seed 由 step 等确定，支持续训。
- `--zo-probe-dtype bf16`：默认与训练精度相同；隐藏状态的差、内积和标量归约在 FP32
  进行。`--zo-probe-dtype fp32` 提供数值诊断路径，关闭该作用域的 autocast/TF32，
  attention 使用原仓库 SDPA fallback；它的实际耗时不应与 BF16 混为一谈。

GPU 验收发现，BF16 下即使编译与 eager 的前向非常接近，两点差分也可能显著放大
舍入差异。因此不能要求两者的 ZO 梯度逐元素接近。默认精度仍保留 BF16；正式实验
前应结合 `check_probe` 检查扰动幅度，FP32 探测路径用于区分量化误差与估计方差。

依据：

1. [MeZO 官方 large_models/mezo.sh](https://github.com/princeton-nlp/MeZO/blob/main/large_models/mezo.sh)
   默认 `EPS=${EPS:-1e-3}`。
2. [MeZO, NeurIPS 2023, §2 / Algorithm 1](https://proceedings.neurips.cc/paper_files/paper/2023/file/a627810151be4d13f907ac898ff7e948-Paper-Conference.pdf)
   使用高斯两点估计，默认一个方向；随机种子可用于重建方向。

这些是微调设置的参考，尚不是本实验从头训练/局部目标上的最优值。上实验机后，先在
非初始 checkpoint 对比 epsilon=3e-4、1e-3、3e-3 的局部方向导数与精确 VJP；
区分有限差分误差、BF16 量化误差、单方向估计方差。lambda=1 按要求保持默认。

## 执行顺序和显存

默认 `none`，每个 microbatch：

1. prelude 正常前向，保留其通向末轮 injection 的图。
2. 早期循环 `no_grad`，仅保留末端状态。
3. 末尾循环和 coda 正常 FO，取得边界梯度；反向完成后释放图。
4. 局部正负探测均 `no_grad`，逐 block 构造临时扰动参数；原参数不进行原地加减。
5. 通过同一随机种子逐参数重建方向，把 ZO 累加到 `.grad`。
6. 累积完成后，由原 `DistMuonAdamW` 平均全部 FO+ZO 梯度并更新一次。

其他三种先将 `e` 作为独立叶子传给末尾 FO，得到 `a,v` 并释放末尾图；完成局部探测后，
再将合并的输入梯度传回 prelude。`separate/joint` 在此期间保留 prelude 图；
`joint_recompute` 在此期间不保存 prelude 的内部激活，最后重新执行其前向。
这些方案都无需 `retain_graph=True`，也不对早期循环构建反向图。
早期 `no_grad` 循环单独关闭 autocast 权重缓存，防止它缓存不带梯度的 BF16 权重，
导致后续共享层 FO 复用缓存而丢失参数梯度；末尾 FO 继续使用正常 autocast 缓存。

这一顺序避免同时保留末尾 FO 激活与探测临时参数。使用逐 block
[`functional_call`](https://docs.pytorch.org/docs/2.9/generated/torch.func.functional_call.html)
避免保存整个共享模块的参数备份，也避免原地 `+epsilon,-2epsilon,+epsilon`
产生的浮点恢复漂移。每次最多临时持有一个 block 的扰动参数，方向也不整模型常驻。
优化器状态和 FO 梯度仍需保存，显存收益主要来自早期循环的激活。

默认 219M、两轮循环、末轮 FO、每方向一对探测、一个方向下，按原仓库的主要矩阵乘/
attention FLOPs 估计：

| 训练方案 | FLOPs / token | 相对完整 FO |
| --- | ---: | ---: |
| 完整 FO | 2.236 G | 100% |
| truncated | 1.690 G | 75.6% |
| Hybrid `none` | 2.236 G | 100% |
| Hybrid `separate` | 2.781 G | 124.4% |
| Hybrid `joint` | 2.236 G | 100% |
| Hybrid `joint_recompute` | 2.303 G | 103.0% |

`none/joint` 的两次早期前向抵消省掉的早期反向；`separate` 多一对输入探测；
`joint_recompute` 多一次两层 prelude 前向。重算版减少 prelude 图的保留时间，
但仍需保存边界张量、输入梯度等，实际峰值显存以测量为准。这个估计不含随机数生成、
参数临时副本、归一化、优化器和通信；**不保证实际 wall time 相同**。
日志分别记录估计 FLOPs、step 时间、tokens/s、各卡最大 allocated/reserved memory。
补回 prelude 的方案还记录输入方向系数、估计输入梯度范数与直接输入梯度范数；
这些诊断量已含梯度累积缩放，输入 ZO 范数尚未乘 lambda。其他分支对应字段为 `null`。
第一步包含编译/优化器状态初始化，不应用来判断稳定吞吐量。

## 启动（实验机）

使用原仓库训练环境；本目录不自动安装依赖、不自动下载数据、不提交集群任务。
已有的预打包数据由 `--data-dir` 指定，默认使用
`$NANOCHAT_BASE_DIR/prepacked_T2048_llama`。真实数据启动会检查序列长度和 tokenizer
元数据，不会默默使用不匹配的 token IDs。

```bash
# 在仓库根目录；PYTHON_BIN 可指向实验机已有的 Python 环境
export PYTHON_BIN=/path/to/env/bin/python
export NANOCHAT_BASE_DIR=/path/to/nanochat-data
export NPROC_PER_NODE=8

bash hybrid/scripts/full_fo.sh
bash hybrid/scripts/truncated.sh
bash hybrid/scripts/hybrid.sh --zo-epsilon 1e-3 --zo-lambda 1

# 或顺序执行三组，分别写入不同的默认输出目录
bash hybrid/scripts/run_all.sh

# 四种 Hybrid prelude 方案；自动使用不同的默认输出目录
bash hybrid/scripts/run_prelude_modes.sh
# 或单独运行，并分别调整参数/输入扰动
bash hybrid/scripts/hybrid_joint_recompute.sh \
    --zo-epsilon 1e-3 --zo-rho 1 --zo-lambda 1

# 调整交叉噪声的相对尺度，使用单独目录
bash hybrid/scripts/hybrid_joint.sh --zo-rho 0.1 \
    --output-dir /path/to/joint-rho-0.1
```

四个独立入口为 `hybrid_none.sh`、`hybrid_separate.sh`、`hybrid_joint.sh`、
`hybrid_joint_recompute.sh`。原 `hybrid.sh` 默认等同于 `hybrid_none.sh`。
原输出目录名保持不变，新方案在目录名中增加 `_prelude_<mode>`。
续训会检查 prelude 方案、参数 epsilon 和 rho，禁止无意间切换实验配置。

默认输出：`$NANOCHAT_BASE_DIR/hybrid_runs/loop2_s16_<mode>_seed42/`。
包含 `metrics.jsonl`、解析后的配置、模型、每卡优化器状态和每卡数据游标。
`--output-dir` 可以覆盖单个模式的目录；不要给 `run_all.sh` 传同一个 output-dir。
目录非空时拒绝覆盖；不要在不同 seed/超参运行中复用它。

```bash
# 不需要 PyTorch / GPU / 数据的配置检查
WORLD_SIZE=8 python3 -m hybrid.train --mode hybrid --dry-run

# 先跑 100 步：仍按原定 19523 步 horizon 计算 LR 和日程
bash hybrid/scripts/hybrid.sh --stop-after 100
# 继续到完整训练终点，保持相同配置、GPU 数和 microbatch
bash hybrid/scripts/hybrid.sh --resume-step 100

# 固定一个新的短训练 horizon（会按原 HyperP 公式重新计算 LR）
bash hybrid/scripts/hybrid.sh --num-iterations 1000 --output-dir /path/to/new-run
```

`--save-every 1000` 可增加定期 checkpoint；默认沿用只在结束时保存的行为，
包括 `--stop-after` 提前结束。没有修改原始 `base_train.py`，使用原入口不会启用
新模式。评估先集中于 held-out loss；未自动启动 CORE、生成或 scaling-law sweep。

## 验收

Mac 仅执行配置/语法检查。2026-11-17 已在 hd-4 的 8 张 RTX 4090 上完成验收：
57 项 CPU 测试、12 项 CUDA 测试、四种 Hybrid 的两卡更新检查，以及六种方案各
4 步的 8 卡小模型 smoke；实际 219M 尺寸的单批检查也通过。
环境、数据版本与检查边界见 [`HD4.md`](HD4.md)。复现检查：

```bash
python -m unittest hybrid.tests.test_config
python -m pytest hybrid/tests -q
# GPU smoke：完整 FO、truncated、四种 Hybrid 各跑 4 步随机数据，非正式实验
NPROC_PER_NODE=1 bash hybrid/scripts/smoke.sh
NPROC_PER_NODE=2 bash hybrid/scripts/smoke.sh
# 验证四种 Hybrid 的两卡混合梯度平均/更新与参数一致性
torchrun --standalone --nproc_per_node=2 -m hybrid.tests.distributed_check

# 在非初始 checkpoint 检查有限差分误差（单 GPU）
python -m hybrid.check_probe --checkpoint-dir /path/to/run --step 100 \
    --epsilons 3e-4 1e-3 3e-3
# 检查输入 ZO；联合方向检查使用 --probe-kind joint
python -m hybrid.check_probe --checkpoint-dir /path/to/run --step 100 \
    --probe-kind input --epsilons 3e-4 1e-3 3e-3
```

数值测试覆盖：与上游完整/截断梯度一致、局部精确 VJP 等于共享模块缺失梯度、
补回精确输入 VJP 后恢复整个模型的完整 FO 梯度、输入/联合有限差分、两种 epsilon
的分母与 lambda、四种方案的 lambda=0 和梯度累积、joint 保留/重算的梯度及优化器
更新一致性、原参数/RNG 不变、异常恢复、prelude 只反传一次、FO 后探测顺序、
CUDA BF16 编译路径、四种方案的 checkpoint 续训一致性。跨卡梯度只由原优化器
同步一次，不额外包 DDP。
GPU 测试另外验证早期 `no_grad` 不丢失共享层 FO 梯度、FP32 参数/联合探测与精确局部
VJP 一致，以及两种 joint 执行方式的一致性。多卡更新参考使用与分片相同的 Muon
矩阵批大小，避免把 BF16 批矩阵乘内核的舍入差异误判为梯度同步错误。

`check_probe` 先用一次精确局部反传取得参考方向导数，再比较 BF16/FP32 探测。
它有意使用早期循环的激活图，只用于验收，不参与训练。FP32 探测与 BF16 参考反传
之间也会存在精度差异；方向导数很接近零时优先看绝对误差。
`--probe-kind parameter`（默认）比较参数方向导数；`input` 比较输入方向导数；
`joint` 比较半差分与 `epsilon_theta*<g_theta,z> + epsilon_e*<g_e,u>`，
不能把联合标量直接与纯参数方向导数比较。`joint` 的输入 epsilon 默认按 checkpoint
记录的 rho 随参数 epsilon 变化；诊断时可用 `--zo-rho` 改比例，或 `--input-epsilon`
固定绝对输入尺度，两者不能同时指定。

### 已训练 checkpoint 的梯度分解

`diagnose_gradients` 使用原 world size、每卡数据游标和梯度累积次数，重放一个
完整 optimizer batch。可读取完整 FO、truncated 或 `hybrid/none` checkpoint，
统一评估末轮 FO 加首轮参数 ZO；每个 microbatch 一个方向：

```bash
torchrun --standalone --nproc_per_node=8 -m hybrid.diagnose_gradients \
    --checkpoint-dir /path/to/hybrid_none/run --step 8000 \
    --output-dir /path/to/diagnostics/eager
# 复核生产编译路径，省略精度扫描和优化器反事实更新
torchrun --standalone --nproc_per_node=8 -m hybrid.diagnose_gradients \
    --checkpoint-dir /path/to/hybrid_none/run --step 8000 \
    --output-dir /path/to/diagnostics/compiled --compile --no-scan --no-updates
# 比较不同训练阶段时固定数据游标和扰动种子（checkpoint 均位于同一 run）
torchrun --standalone --nproc_per_node=8 -m hybrid.diagnose_gradients \
    --checkpoint-dir /path/to/full_fo/run --step 16000 \
    --data-step 8000 --direction-step 8000 \
    --output-dir /path/to/diagnostics/full_fo_16000 --compile --no-scan --no-updates
```

输出分别记录 suffix FO、精确早期参数 VJP、实际 BF16 ZO、用精确方向导数构造的
随机方向估计，以及 FP32 ZO，包含每卡与平均后范数、夹角和高斯估计器方差预测。
epsilon 扫描分别使用对应 BF16/FP32 局部映射的自动微分参考，避免混淆精度差异。
默认数据游标和扰动种子取 `--step`；`--data-step`、`--direction-step` 可分别覆盖。
结果记录 checkpoint 的训练模式、精确早期/末轮梯度的夹角及各 rank 实际诊断数据
的 SHA256，用于核对跨 checkpoint 比较是否使用相同 batch。

优化器对照每次恢复同一份权重和优化器状态，另提供清零历史矩的对照；更新仅在
内存中进行，不保存模型、不恢复正式训练。这里的 `oracle_early_parameters` 只补回
共享参数的精确早期梯度，prelude 仍使用直接 FO，因此不等同于整个模型的完整 BPTT。
这些单步对照用于定位梯度/更新方向，不能当作不同 lambda 的收敛对比。

### 从完整 FO checkpoint 测试范数匹配

`norm_match` 是独立的短跑入口：恢复同一份完整 FO 权重、分片优化器状态和数据
游标，依次运行完整 FO、truncated、若干范数匹配 Hybrid，以及原始 lambda=1
Hybrid。各分支使用原训练 horizon 的学习率日程，默认各跑 200 步，验证数据相同。
当前限于两轮循环、末轮 FO、`prelude_mode=none` 和原 world size。

```bash
# 单批梯度及实际优化器更新方向诊断，不保存权重
torchrun --standalone --nproc_per_node=8 -m hybrid.diagnose_gradients \
    --checkpoint-dir /path/to/full_fo/run --step 8000 \
    --output-dir /path/to/norm_match/diagnostic --compile --no-scan \
    --match-ratios 1 0.3 0.1

# 六个独立分支，各训练 200 步；输出目录必须为空
torchrun --standalone --nproc_per_node=8 -m hybrid.norm_match \
    --checkpoint-dir /path/to/full_fo/run --step 8000 --steps 200 \
    --output-dir /path/to/norm_match/short_runs --ratios 1 0.3 0.1
```

对共享参数，先分别累积并跨卡平均末轮 FO 与首轮 ZO，再计算全局 L2 范数：

```text
effective_lambda = alpha * ||g_FO|| / ||g_ZO||
g_mixed = g_FO + effective_lambda * g_ZO
```

`--ratios` 指定 alpha，即缩放后 ZO 范数与 FO 范数的目标比值；不是固定的
原始 `zo_lambda`。每步动态计算系数，不逐层归一化。共享参数以外保留 suffix
FO，prelude 的首轮间接梯度不补回。优化器仍为原 MuonH + AdamW；没有增加裁剪。
范数匹配改变了估计器的统计性质，不能再把缩放后的量称为原首轮梯度的无偏估计。

输出包含每个分支的 `metrics.jsonl`、首批逐 rank SHA256、末步 checkpoint，
以及总目录的 `manifest.json` 和 `summary.json`。各分支检查初始验证 loss 一致。
为测量方便，额外保留 ZO 梯度缓冲区并同步共享梯度；此入口不用于显存/吞吐比较。
范数匹配 checkpoint 带有专门元数据，普通 `hybrid.train` 会拒绝直接续训，以免
无意中丢失动态缩放规则。此短跑入口本身尚不支持续训。

### 从零预训练的范数匹配

正式入口使用相同的范数匹配实现，并将 `zo_norm_ratio` 写入配置与 checkpoint：

```bash
# 仍沿用 19,523 步原日程；每 250 步验证、每 2,000 步保存
bash hybrid/scripts/hybrid_norm_match.sh 1 --save-every 2000 --output-dir /path/to/norm1
bash hybrid/scripts/hybrid_norm_match.sh 0.3 --save-every 2000 --output-dir /path/to/norm03
bash hybrid/scripts/hybrid_norm_match.sh 0.1 --save-every 2000 --output-dir /path/to/norm01

# 从正式入口产生的 checkpoint 恢复，比例必须保持一致
bash hybrid/scripts/hybrid_norm_match.sh 1 --save-every 2000 \
    --output-dir /path/to/norm1 --resume-step 2000
```

对应参数为 `--mode hybrid --prelude-mode none --zo-lambda 1 --zo-norm-ratio ALPHA`。
默认 `zo_norm_ratio=None`，旧模式和旧 checkpoint 的行为不变。启用后要求原始
`zo_lambda=1`，避免固定权重和目标范数产生歧义。每步日志增加 FO/原始 ZO/缩放后
ZO 的共享参数范数、FO–ZO cosine 和 `effective_lambda`。从零初始化时边界梯度
为零，ZO 为零，此时缩放系数定义为零。

两个入口均先平均 FO/ZO，再匹配全体共享参数的范数。额外 ZO 缓冲区和通信仍然
存在，尚未做显存或通信优化。`grad_norm_local_max` 是混合后各卡梯度缓冲区范数
的最大值；启用匹配时，共享部分已经跨卡平均，其余参数仍由优化器平均。

### 从完整 FO 渐进切换，并继续到原训练终点

`fo_transition.sh` 从指定完整 FO checkpoint 恢复权重、分片优化器、RNG 和数据游标，
在新输出目录训练。默认从 8,000 步开始，用 1,000 次更新过渡，仍在原定 19,523 步
结束；学习率日程不重启。完整 FO 原训练曲线作为参考，另运行渐变 truncated 和
三个渐变 norm ZO 分支：

```bash
bash hybrid/scripts/fo_transition.sh truncated /path/to/full_fo/run \
    --save-every 2000 --output-dir /path/to/fo_to_truncated
bash hybrid/scripts/fo_transition.sh 1 /path/to/full_fo/run \
    --save-every 2000 --output-dir /path/to/fo_to_norm1
# 将第一个参数改成 0.3 / 0.1，分别启动另外两组。

# 从新分支自己的 checkpoint 恢复；原 fork 参数和 transition_steps 必须相同。
bash hybrid/scripts/fo_transition.sh 1 /path/to/full_fo/run \
    --save-every 2000 --output-dir /path/to/fo_to_norm1 --resume-step 10000
```

可用 `--fo-checkpoint-step`、`--transition-steps` 覆盖默认值。过渡长度 N≥2 时，
第一个恢复更新的 s=0，最后一个过渡更新的 s=1：默认第 8,001 步更新完全为 FO，
第 9,000 步更新已经完全使用目标梯度。`--transition-steps 0` 表示立即切换。

对共享参数，记 f 为末轮精确梯度、h 为首轮精确梯度、z 为首轮局部 ZO：

```text
truncated:  g = f + (1-s) * h
norm ZO:    g = f + (1-s) * h + s * alpha * ||f|| / ||z|| * z
```

范数基准始终是跨卡平均后的纯末轮 FO；保留的首轮精确梯度不进入分母或基准。
prelude/embedding 的首轮间接梯度也乘 (1-s)，直接路径梯度保留。这样整个模型
在 s=0 时等于完整 FO，在 s=1 时等于原 truncated 或 `hybrid/none` 范数匹配模式。
各项先混合，再由同一个原优化器更新一次。

中间阶段使用固定边界 cotangent 重算首轮精确 VJP，并重算 prelude 来合并直接和
间接输入梯度；另保留一个共享参数精确梯度缓冲区。该阶段计算量会增加，日志的
FLOPs 估计计入重算和首轮 FO 部分。过渡结束后释放精确梯度缓冲区，恢复目标模式
的执行路径。checkpoint 保存固定的过渡起点与长度，恢复时不会重新开始渐变。

验收覆盖：全模型梯度与完整 FO/目标梯度凸组合一致（含 prelude）、两端路径、
梯度累积、原 checkpoint 不变，以及跨过过渡终点的断点续训一致性。

### 配对复现与训练故障记录

独立种子的复现应同时重跑 truncated 对照。以下两组初始化种子均为 43，数据
顺序保持原固定 parquet 顺序；ZO 方向种子改为 12346。其余配置保留原日程：

```bash
bash hybrid/scripts/launch.sh truncated --seed 43 --zo-seed 12346 \
    --save-every 2000 --output-dir /path/to/seed43_truncated
bash hybrid/scripts/hybrid_norm_match.sh 0.1 --seed 43 --zo-seed 12346 \
    --save-every 2000 --output-dir /path/to/seed43_norm01
```

单个 rank 出错时，训练入口先打印原始异常，再退出，由 torchrun 终止其他进程；
失败路径不调用需要其他 rank 共同参与的 NCCL 清理。正常结束仍执行原清理流程。
这避免将显存不足等原始错误掩盖成十分钟后的通信超时。

可选的卡住诊断只记录调用栈，不改变梯度和训练配置：

```bash
export HYBRID_STALL_SECONDS=180
export HYBRID_DIAGNOSTICS_DIR=/path/to/run_diagnostics
```

成功更新、验证或保存后会重置计时；超过设定时间无进展时，各 rank 在独立文件
记录 Python 调用栈。目录应放在训练输出目录外，以免触发非空输出目录保护。
默认关闭。两卡错误退出回归检查：

```bash
python -m hybrid.tests.distributed_failure_check --output-dir /path/to/failure_check
```

该检查故意让 rank 0 报错，要求在 120 秒内退出并保留原异常，且不进入失败 rank
的 collective cleanup。检查的子训练进程应失败，检查程序本身通过后退出码为 0。

### 等待 GPU 空闲后运行

`hybrid.wait_for_gpus` 默认每 180 秒查询一次指定 GPU，连续两次同时满足以下
条件才启动命令：无计算进程、每卡已用显存不超过 256 MiB、利用率不超过 5%。
任一 GPU 不满足条件即重置连续计数；查询失败、超时、缺卡或指标为 N/A 时继续
等待。命令通过检查到的 GPU UUID 绑定物理卡。文件锁防止本队列重复启动。

```bash
python -m hybrid.wait_for_gpus \
    --state-file /path/to/gpu_wait.json --lock-file /path/to/shared_queue.lock \
    -- bash hybrid/scripts/hybrid_norm_match.sh 0.1 --seed 43 --zo-seed 12346 \
       --save-every 2000 --output-dir /path/to/seed43_norm01
```

`hybrid.experiment_queue --experiment-dir DIR` 按 DIR/manifest.json 顺序运行实验，
每个实验启动前都使用相同空闲检查；一个实验正常结束且最终 checkpoint 完整后
才启动下一个，失败则停止队列。轮询状态保存在分支的 `gpu_wait.json` 和总目录
的 `status.json`。状态、日志与训练输出均可放在数据盘。

空闲检查不是机器级 GPU 预约；其他账户仍可能在检查后启动任务。脚本不会结束
其他用户的进程，也不会在失败后自动跳过实验或修改训练超参数。
