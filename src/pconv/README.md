# PConv-LoRA 对比实验流程说明

本模块用于在当前古画修复项目中加入 **Partial Convolution（PConv）** 对比实验。整体思路是：保持原仓库的 `clean / damaged / mask` 数据格式不变，在预训练 PConv 模型基础上进行 LoRA 微调，然后与 SD Inpainting / SD-LoRA 的结果进行统一评价。

## 1. 模块位置

PConv 相关代码放在：

```text
src/pconv/
  __init__.py
  partialconv2d.py
  pconv_unet.py
  lora_conv.py
  datasets.py
  train_pconv_lora.py
  infer_pconv_lora.py
  utils.py
```

各文件作用：

| 文件 | 作用 |
|---|---|
| `partialconv2d.py` | Partial Convolution 基础层 |
| `pconv_unet.py` | PConv-UNet 网络结构 |
| `lora_conv.py` | 给卷积层加入 LoRA 分支 |
| `datasets.py` | 读取 `clean / damaged / mask` 数据 |
| `train_pconv_lora.py` | PConv-LoRA 微调入口 |
| `infer_pconv_lora.py` | PConv-LoRA 推理入口 |
| `utils.py` | checkpoint、保存图像、loss 等工具函数 |

## 2. 数据格式

本模块沿用原仓库的数据结构：

```text
data/
  train/
    clean/
    damaged/
    mask/

  val/
    clean/
    damaged/
    mask/
```

要求三类图像文件名对应，例如：

```text
data/train/clean/0001.png
data/train/damaged/0001.png
data/train/mask/0001.png
```

训练时：

```text
输入：damaged + mask
目标：clean
```

验证时：

```text
输入：data/val/damaged + data/val/mask
目标：data/val/clean
```

## 3. Mask 约定

原仓库 mask 约定：

```text
255：需要修复区域
0：正常区域
```

PConv 内部需要的是 valid mask：

```text
1：有效区域
0：缺失区域
```

因此代码中会自动进行转换：

```text
damage_mask = mask / 255
valid_mask = 1 - damage_mask
```

不需要手动修改 mask。

## 4. 目录准备

建议使用如下目录结构：

```text
models/
  pconv/
    pconv.pth

checkpoints/
  pconv_lora/

outputs/
  pconv_lora/
  metrics/
  vis/
```

创建目录：

```bash
mkdir -p models/pconv
mkdir -p checkpoints/pconv_lora
mkdir -p outputs/pconv_lora
mkdir -p outputs/metrics
mkdir -p outputs/vis/pconv_lora
```

## 5. 预训练权重

PConv-LoRA 微调需要先准备 PConv 预训练权重：

```text
models/pconv/pconv.pth
```

可以从公开 PConv PyTorch 复现仓库的 Release 或 Google Drive 权重中获取，然后放到上述路径并重命名为 `pconv.pth`。

注意：

- 该权重必须是 PConv / PConv-UNet 类型权重；
- 不能使用 Stable Diffusion 的权重；
- 如果出现 `Missing key(s)`、`Unexpected key(s)` 或 `size mismatch`，说明权重结构与当前 `PConvUNet` 不完全匹配，需要做 checkpoint key 映射或调整模型结构。

检查权重是否存在：

```bash
ls models/pconv/pconv.pth
```

## 6. Smoke Test

正式训练前建议先跑 1 轮，确认数据、模型和权重都能正常加载：

```bash
python -m src.pconv.train_pconv_lora \
  --train_root data/train \
  --val_root data/val \
  --pretrained_pconv models/pconv/pconv.pth \
  --output_dir checkpoints/pconv_lora_smoke \
  --resolution 512 \
  --batch_size 1 \
  --epochs 1 \
  --rank 4 \
  --learning_rate 1e-4
```

如果运行成功，会生成：

```text
checkpoints/pconv_lora_smoke/
  best.pth
  latest.pth
  history.json
```

## 7. 正式微调

推荐先使用 10 个 epoch 进行正式实验：

```bash
python -m src.pconv.train_pconv_lora \
  --train_root data/train \
  --val_root data/val \
  --pretrained_pconv models/pconv/pconv.pth \
  --output_dir checkpoints/pconv_lora \
  --resolution 512 \
  --batch_size 1 \
  --epochs 10 \
  --rank 8 \
  --learning_rate 1e-4
```

训练完成后输出：

```text
checkpoints/pconv_lora/
  best.pth
  latest.pth
  history.json
```

其中：

| 文件 | 含义 |
|---|---|
| `best.pth` | 验证集 loss 最优的权重 |
| `latest.pth` | 最后一轮训练后的权重 |
| `history.json` | 训练和验证 loss 记录 |

如果显存充足，可以将 `--batch_size 1` 改为 `--batch_size 2`。

## 8. 推理

使用微调后的 `best.pth` 在验证集上推理：

```bash
python -m src.pconv.infer_pconv_lora \
  --input_dir data/val/damaged \
  --mask_dir data/val/mask \
  --checkpoint checkpoints/pconv_lora/best.pth \
  --output_dir outputs/pconv_lora \
  --resolution 512
```

推理结果会保存到：

```text
outputs/pconv_lora/
```

输出图像文件名会尽量与输入图像保持一致，方便后续统一计算指标。

## 9. 指标计算与样本可视化

使用统一的 `src.metrics` 计算指标：

```bash
python -m src.metrics \
  --pred_dir outputs/pconv_lora \
  --clean_dir data/val/clean \
  --mask_dir data/val/mask \
  --damaged_dir data/val/damaged \
  --output outputs/metrics/pconv_lora_metrics.json \
  --detail_csv outputs/metrics/pconv_lora_metrics_detail.csv \
  --vis_dir outputs/vis/pconv_lora \
  --top_k 5
```

输出：

```text
outputs/metrics/
  pconv_lora_metrics.json
  pconv_lora_metrics_detail.csv

outputs/vis/pconv_lora/
  best/
  middle/
  worst/
```

其中 CSV 包含：

```text
filename,psnr,ssim,masked_psnr,mask_ratio
```

## 10. 与 SD-LoRA 结果对比

如果已经完成 SD-LoRA 的指标计算，并得到：

```text
outputs/metrics/sd_lora_metrics_detail.csv
outputs/metrics/pconv_lora_metrics_detail.csv
```

可以统一生成样本对比：

```bash
python -m src.metrics \
  --compare_csv SD-LoRA=outputs/metrics/sd_lora_metrics_detail.csv \
  --compare_csv PConv-LoRA=outputs/metrics/pconv_lora_metrics_detail.csv \
  --compare_pred_dir SD-LoRA=outputs/sd_lora \
  --compare_pred_dir PConv-LoRA=outputs/pconv_lora \
  --clean_dir data/val/clean \
  --damaged_dir data/val/damaged \
  --mask_dir data/val/mask \
  --vis_dir outputs/vis/compare \
  --top_k 5
```

输出：

```text
outputs/vis/compare/
  SD-LoRA/
    best/
    middle/
    worst/

  PConv-LoRA/
    best/
    middle/
    worst/
```

## 11. 推荐完整流程

```text
准备 PConv 预训练权重
  ↓
Smoke Test 跑通 1 轮
  ↓
正式微调 PConv-LoRA
  ↓
使用 best.pth 进行验证集推理
  ↓
计算 PSNR / SSIM / masked PSNR
  ↓
与 SD-LoRA 结果进行对比
```

对应命令顺序：

```bash
python -m src.pconv.train_pconv_lora \
  --train_root data/train \
  --val_root data/val \
  --pretrained_pconv models/pconv/pconv.pth \
  --output_dir checkpoints/pconv_lora_smoke \
  --resolution 512 \
  --batch_size 1 \
  --epochs 1 \
  --rank 4 \
  --learning_rate 1e-4

python -m src.pconv.train_pconv_lora \
  --train_root data/train \
  --val_root data/val \
  --pretrained_pconv models/pconv/pconv.pth \
  --output_dir checkpoints/pconv_lora \
  --resolution 512 \
  --batch_size 1 \
  --epochs 10 \
  --rank 8 \
  --learning_rate 1e-4

python -m src.pconv.infer_pconv_lora \
  --input_dir data/val/damaged \
  --mask_dir data/val/mask \
  --checkpoint checkpoints/pconv_lora/best.pth \
  --output_dir outputs/pconv_lora \
  --resolution 512

python -m src.metrics \
  --pred_dir outputs/pconv_lora \
  --clean_dir data/val/clean \
  --mask_dir data/val/mask \
  --damaged_dir data/val/damaged \
  --output outputs/metrics/pconv_lora_metrics.json \
  --detail_csv outputs/metrics/pconv_lora_metrics_detail.csv \
  --vis_dir outputs/vis/pconv_lora \
  --top_k 5
```

## 12. 常见问题

### 1. PConv 权重和 SD 权重能混用吗？

不能。PConv 是 CNN / U-Net 类结构，Stable Diffusion 是扩散模型，两者结构完全不同，权重不能混用。

### 2. 为什么要先跑 smoke test？

smoke test 可以快速检查路径、数据格式、checkpoint 加载和训练流程是否正常，避免正式训练跑很久后才发现错误。

### 3. 为什么训练时用 `damaged + mask`，而不是只用 damaged？

PConv 需要 mask 来判断哪些像素是有效区域、哪些像素是缺失区域。没有 mask 时，模型无法明确知道需要修复的位置。

### 4. 为什么 PConv 的 mask 要反转？

原仓库中 `255` 表示需要修复区域，而 PConv 中 `1` 表示有效区域。因此代码内部会将原始 mask 转换为：

```text
valid_mask = 1 - damage_mask
```

### 5. 如果显存不足怎么办？

优先调整：

```bash
--batch_size 1
```

如果仍然不足，可以降低分辨率：

```bash
--resolution 256
```

但如果要和 SD-LoRA 进行公平比较，建议尽量保持相同分辨率。
