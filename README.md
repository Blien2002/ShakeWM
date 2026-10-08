# ShakeWM

历史 RGB 与已交付 IMU 条件下的被动物体响应预测。冻结 V-JEPA 2.1 单图特征，自建块因果预测器；不接入策略动作或机器人/物体真值状态。

当前交付：训练、固定起点评测、冻结特征缓存、保存恢复、CPU 单元测试和 synthetic smoke。**尚未跑真实预训练权重、真实采集数据或 GPU 验收；synthetic 结果不证明 IMU 收益。** 正式模型与 smoke 模型使用同一实现，配置分开。

## 安装

优先使用已有 PyTorch 环境，避免在空间不足的根盘重新安装 CUDA。

```bash
git clone https://github.com/Blien2002/ShakeWM.git
cd ShakeWM
python -m pip install --no-deps --no-build-isolation -e .
python -c 'import torch, numpy; print(torch.__version__, numpy.__version__)'
# 缺少测试依赖时：python -m pip install pytest
# 仅真实 encoder 需要：python -m pip install 'timm>=1.0,<2'
```

依赖为 PyTorch >=2.5、NumPy >=1.24；pytest >=8 用于测试。基线 CPU 验证环境为 Python 3.13 / PyTorch 2.9.1 / NumPy 1.26.4，精确实测版本见 [CPU 报告](docs/CPU_TEST_REPORT.md)。保留 editable checkout，官方 encoder 从仓库内固定来源的最小源码集加载；不支持把单独 wheel 拷走后丢弃 `third_party`。

## 五分钟 CPU 闭环

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
bash scripts/smoke.sh runs/my-cpu-smoke
```

脚本生成明确标记为 synthetic 的 4 条小回合，完成缓存、V0 训练及恢复、独立 RGB-only、V1 训练/评测和在线 mock 编码。只使用 CPU，不下载权重。输出目录必须不存在，避免覆盖。`PYTHON=/path/to/python bash scripts/smoke.sh runs/another-smoke` 可指定环境。

若要分步执行：

```bash
python -m shakewm.cli fixtures --output data/toy --seconds 4
python -m shakewm.cli fit-norm --manifest data/toy/manifest.json --output data/toy/norm.json
python -m shakewm.cli cache --config configs/smoke.json --manifest data/toy/manifest.json --output cache/toy --encoder mock --device cpu
python -m shakewm.cli train --config configs/smoke.json --manifest data/toy/manifest.json --normalization data/toy/norm.json --cache cache/toy --output runs/toy --stop-after 1 --device cpu
python -m shakewm.cli train --config configs/smoke.json --manifest data/toy/manifest.json --normalization data/toy/norm.json --cache cache/toy --output runs/toy --resume runs/toy/latest.pt --device cpu
python -m shakewm.cli eval --config configs/smoke.json --manifest data/toy/manifest.json --normalization data/toy/norm.json --cache cache/toy --checkpoint runs/toy/latest.pt --output runs/toy/test.json --device cpu
```

`--stop-after` 只提前停止，不改变总步数及余弦调度。恢复必须使用原配置；checkpoint 包含模型、AdamW、scheduler、Python/NumPy/Torch/sampler RNG、global step、归一化、split hash 和 teacher 契约。CUDA 运行另存 CUDA RNG。日志为 JSONL，评测为 JSON。

## 正式配置与信息边界

| 项目 | 默认值 |
| --- | --- |
| teacher | 官方蒸馏 `vjepa2_1_vit_base_384` / `ema_encoder`，冻结、eval、no_grad |
| 图像 | 原始单第三人称 256×256 RGB；每张单独走 T=1 图像分支；末层归一化特征 256×768 |
| predictor | 16 层，宽 1024，16 heads，MLP ratio 4，pre-norm，视觉 768↔1024 |
| token | 每块 `[短 IMU, 长 IMU, 256 patches]` 共 258；两 IMU 槽始终保留 |
| 位置/注意力 | 视觉时间/行/列 3D RoPE；IMU 只旋转时间通道；同块全可见、禁止未来块 |
| 时间 | RGB 采集 20 Hz，训练 10 Hz；默认 C=10、H=10，支持修改 C=10–16、H=1–10 |
| V0 / V1 / RGB-only | `configs/v0.json` / `v1.json` / `rgb_only.json` |
| loss | 有效转移全图 feature L1；TF 与 rollout 独立前向及 backward |
| rollout | 历史一次 prefill，未来两 IMU 槽 NO-IMU；预测回填，不读未来真视觉/IMU |
| TBPTT | 每 4 步及时 backward，再 detach 预测与所有 K/V；所有段结束后才 optimizer step |
| optimizer | AdamW，1e-4，weight decay .04，1000-step warmup / cosine，BF16 可选 |

每层按完整时间块调用 SDPA，`is_causal=False` 的查询只得到当前及之前块的 K/V；不构造完整序列注意力矩阵，也不把块语义改为逐 token 因果。cache 仅存在于当前 rollout，最多 C+H−1 块。C16/H10 为 6450 token。正式配置开启 non-reentrant activation checkpoint。

短 IMU：20×6，三层因果卷积（5，dilation 1/2/4，64/128/128）加四头单 query 池化。长 IMU：600×6，63-tap 因果 FIR（20 Hz cutoff，155 ms 群延迟）、stride 4、6 个双卷积残差 TCN（dilation 1…32），池化至 128。左 padding 是网络边界处理，不计作真实观测。生产数据必须满足每个历史步的 live 窗口资格。V0 长槽、RGB-only 两槽均为可学习 NO-IMU；样本级联合 IMU dropout=.25。

## 官方 encoder 与缓存

来源固定为 [facebookresearch/vjepa2@204698b](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812)。[upstream.lock.json](upstream.lock.json) 保存 commit、官方 checkpoint URL 及每份未修改源码的 SHA256；加载时核验源码。官方 checkpoint 表称约 80M，所实例化完整 encoder 实际为 86,833,152 参数。

官方发布输入尺寸为 384，本工程保持该模型参数配置，通过官方 `interpolate_rope=True` 接受 256 单图，得到 16×16 网格。当前固定提交的 Hub factory 内下载基址为 localhost，故本工程**不调用其自动下载路径**，直接构造相同官方 encoder，显式加载本地 `ema_encoder`，仅去除已知 `module.` / `backbone.` 前缀，然后 `strict=True`。不实例化官方 masked predictor。

权重位置请选容量充足的已有磁盘。`--encoder-checkpoint` 必填，不会隐式下载。先运行真实权重 shape/数值门禁，再缓存：

```bash
python scripts/check_official_encoder.py --checkpoint /mnt/large/models/vjepa2_1_vitb_dist_vitG_384.pt --device cuda:0 --output runs/official-check.json
python -m shakewm.cli fit-norm --manifest /mnt/large/imu_wm/manifest.json --output /mnt/large/imu_wm/train_norm.json
python -m shakewm.cli cache --config configs/v0.json --manifest /mnt/large/imu_wm/manifest.json --encoder official --encoder-checkpoint /mnt/large/models/vjepa2_1_vitb_dist_vitG_384.pt --output /mnt/large/imu_wm/features-v1 --device cuda:0
python -m shakewm.cli train --config configs/v0.json --manifest /mnt/large/imu_wm/manifest.json --normalization /mnt/large/imu_wm/train_norm.json --cache /mnt/large/imu_wm/features-v1 --output /mnt/large/runs/v0 --device cuda:0
python -m shakewm.cli eval --config configs/v0.json --manifest /mnt/large/imu_wm/manifest.json --normalization /mnt/large/imu_wm/train_norm.json --cache /mnt/large/imu_wm/features-v1 --checkpoint /mnt/large/runs/v0/latest.pt --output /mnt/large/runs/v0/test.json --device cuda:0
```

以上 `/mnt/large/...` 为需替换的外部路径，并不代表本机存在。GPU 命令由调度方选择空闲资源后执行。预训练权重可加 `--encoder-sha256 HASH` 校验。只有 CPU 代码检查时用 `--random-architecture`，结果会明确标记未加载预训练权重。

在线编码：训练命令去掉 `--cache`，加 `--encoder official --encoder-checkpoint PATH`。图像仍逐帧编码，历史及目标不经过跨时间 attention。预处理为 RGB `[0,1]`、ImageNet mean/std、严格 256×256、无 crop/额外 LayerNorm。缓存为 FP16 `.npy`，写入时逐 batch，读取时 mmap；在线输出为 FP32，二者存在 FP16 量化差。契约记录代码/权重哈希、视角、采样率、预处理、层、网格和归一化，原始文件与缓存文件校验和也必须匹配。

当前实现是单进程/单设备基础训练，无 DDP 或梯度累积调度。16 层正式 AdamW checkpoint 远大于 smoke，不应保存到仅剩约 1 GiB 的根盘。未做吞吐、24 GB 显存或多卡可行性承诺。

## 数据与评测

[数据接口](docs/DATA_CONTRACT.md) 定义将 `shakebench.imu_wm.v1` 记录转成训练用 NPZ 的边界。当前只提供并验证 normalized NPZ 入口；尚未验证采集端实际文件布局到该格式的转换。不得把现有 smoke 当成正式 0/1350 数据已到齐。

先按 state 分组冻结 train/val/test manifest，再切窗口；同 state 所有场景、seed、视角必须同 split，split 间 seed 池不相交。加载器直接拒绝 state/seed 泄漏。均值/方差只用 train live 数据，不做逐窗口归一化；负时间 reset 预填无效。IMU 以实际 delivery≤当前帧时刻裁切，真实采样间隔、数量和最新可用性均校验；长历史不完整不补伪样本。

每个 horizon 输出 latent L1、copy-last L1、有效样本数，并保留 episode/state/origin 明细。超出离桌或时间上限的目标 mask 掉，无样本时指标为 null。TF 的上下文长度 1…C 会写入日志，与固定起点多步结果分开。

独立 RGB-only 用 `configs/rgb_only.json` 重新训练，模型参数和 token 预算相同。`eval --no-imu` 仅为测试时屏蔽诊断。V0/V1 公平比较时，在两个配置中都设 `data.eligibility="v1"` 并重新训练，保证共同合格窗口；各自覆盖率另报。不要直接比较 smoke 脚本中 V0 全覆盖与 V1 长历史子集的数值。

## 状态与许可

[CPU 测试报告](docs/CPU_TEST_REPORT.md) 与 [上游说明](third_party/NOTICE.md) 列明已验证范围。当前未实现 ROI、物理 probe、slot、pixel decoder、V2 相位分支；这些不属于本轮全图 toy。模型入口没有动作、seed、scenario、GT pose 或未来强制条件。

新代码采用 MIT。上游文件保留原版权及许可证；没有复制其他私有项目，没有上传数据、checkpoint 或凭据。

<!-- native-import-revision -->
## Native recording importer and execution handoff

Direct `shakebench.imu_wm.v1` import is now supported. The normalized episode contract remains the model interface. See [native import and CPU evidence](docs/NATIVE_IMPORT.md), [exact commands](docs/EXECUTION_HANDOFF.md), and [official-checkpoint verification status](docs/GPU_HANDOFF.md). The existing recordings provide train-only mock diagnostics; the pretrained encoder numerical gate remains failed and GPU execution is untested.
