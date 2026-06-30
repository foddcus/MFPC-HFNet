# MFPC-HFNet 公开版训练工程说明

更新时间：2026-06-30  
维护者：ljy / GG  
适用范围：公开版 MFPC-HFNet / 多源土壤属性反演训练代码  
默认目标：SOC  
默认数据库格式：`public_single_npz_v1`

## 0. 快速运行顺序

这份 README 只说明当前公开版工程的使用方式。建议按下面顺序执行：

1. 确认 `Training Code` 和 `PublicSoilSampleDatabase` 是同级目录。
2. 创建 Python 3.12 环境并安装依赖。
3. 检查公开数据库数量和字段。
4. 先执行 `--dry-run` 查看最终配置。
5. 再做 1 个 Fold、1 个 epoch 的冒烟测试。
6. 最后启动正式训练。

最小配置检查命令：

```powershell
Set-Location "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code"
python Train_main.py --dry-run
```

`--dry-run` 只打印最终配置，不启动训练。

## 1. 工程作用

本工程用于训练和导出多源土壤属性反演模型。当前公开版围绕 `PublicSoilSampleDatabase` 单文件样本数据库运行，默认训练目标为 SOC，默认主模型为 `MFPCHFNetV2_Full`。

当前默认训练任务：

```text
训练菜单：mfpchfnetv2
训练模型：MFPCHFNetV2_Full
目标变量：SOC
数据库：Training Code 上一级的 PublicSoilSampleDatabase
交叉验证：8 Fold，运行 8 个 Fold
输出目录：Training Code/ModelData
训练后导出：ONNX
```

## 2. 目录结构

默认代码假定 `Training Code` 和 `PublicSoilSampleDatabase` 是并列目录：

```text
Multisource Data LJY 2025/
  Training Code/
    README.md
    Train_main.py
    Train_config.py
    Train_core.py
    Train_support.py
    Train_optimizer.py
    Train_export_onnx.py
    Data_PublicSampleDatabase.py
    Data_LoaderRuntimeAuto.py
    Data_BuildPcaPriorsFull.py
    Menu_MFPCHFNetV2.py
    Menu_InputAblation.py
    Menu_Compare_AllBackbones.py
    Model_MFPCHFNet.py
    Metrics_core.py
    ModelAssets/
      pca_priors_full.pt
    Tool/
      Tool_CheckTrainingPolicy.py
  PublicSoilSampleDatabase/
    README.md
    public_dataset_manifest.json
    samples/
      00465C9D.npz
      ...
```

本机当前路径示例：

```text
C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code
C:\Users\94562\Desktop\Multisource Data LJY 2025\PublicSoilSampleDatabase
```

注意事项：

| 项目 | 说明 |
| --- | --- |
| 路径空格 | `Training Code` 路径中有空格，PowerShell 命令里必须加英文双引号。 |
| 默认数据库 | 训练入口会自动寻找 `..\PublicSoilSampleDatabase`。 |
| 临时换库 | 数据库在其他位置时，优先用 `--dataset-root` 临时覆盖。 |
| 输出目录 | `ModelData/` 用于保存模型、日志和汇总表，训练后可能很大。 |

## 3. 当前公开数据库结构

公开数据库格式为：

```text
public_single_npz_v1
```

使用者需要关心的结构：

```text
PublicSoilSampleDatabase/
  public_dataset_manifest.json
  samples/
    8位样本ID.npz
```

每个 `.npz` 样本包含：

| 字段 | 含义 | 形状或示例 |
| --- | --- | --- |
| `sample_id` | 样本 ID，与文件名主干一致。 | `00465C9D` |
| `hyper` | HyperVISNIR 光谱特征。 | `(681,)` |
| `nir` | NIR 数值特征。 | `(5,)` |
| `image` | 8 通道图像张量。 | `(8, 1024, 1024)` |
| `labels` | 标签数组。 | `(1,)` |
| `target_names` | 标签名称数组。 | `["SOC"]` |

当前公开库只发布 SOC 标签，因此保持：

```text
TARGET_MODE = "soc"
```

## 4. 主要源码职责

| 类型 | 文件或目录 | 作用 |
| --- | --- | --- |
| 训练入口 | `Train_main.py` | 手工参数面板和终端入口。 |
| 参数装配 | `Train_config.py` | 合并菜单默认值、入口面板值和命令行临时覆盖。 |
| 训练引擎 | `Train_core.py` | 执行 Dataset、Fold、模型、优化器、训练循环、断点和汇总。 |
| 折分/日志工具 | `Train_support.py` | 稳定折分、batch 规划、日志写出等训练支持逻辑。 |
| 优化器策略 | `Train_optimizer.py` | 构建 AdamW 参数组、分组学习率和冻结/解冻策略。 |
| 数据读取 | `Data_PublicSampleDatabase.py`、`Data_LoaderRuntimeAuto.py` | 读取公开数据库、按菜单裁剪输入源、组织 Dataset 条目。 |
| PCA 先验 | `Data_BuildPcaPriorsFull.py` | 训练时按当前 Fold 的 Train 子集构建图像分支 PCA/归一化先验。 |
| 模型结构 | `Model_*.py` | 定义 MFPC-HFNet 和对比模型结构。 |
| 训练菜单 | `Menu_*.py` | 声明模型清单、输入尺寸、batch size 和模型级策略。 |
| 指标输出 | `Metrics_core.py` | 保存指标、预测表、散点图和最终汇总。 |
| ONNX 导出 | `Train_export_onnx.py` | 训练后选择代表 Fold，导出 ONNX 和追溯 JSON。 |
| 工程检查 | `Tool/Tool_CheckTrainingPolicy.py` | 检查训练工程边界。 |

参数优先级固定为：

```text
菜单显式设置 > Train_main.py 面板补充设置 > 训练代码默认
```

命令行参数只作为本次运行的临时覆盖，不会回写源码。

## 5. Python 与 CUDA 环境

推荐使用 Python 3.12。正式训练默认要求 CUDA GPU。

创建虚拟环境：

```powershell
Set-Location "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 提示禁止运行脚本，可在当前窗口临时放开执行策略：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

通用依赖：

```powershell
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn opencv-python openpyxl onnx thop timm
```

PyTorch 需要按本机显卡、驱动和 CUDA 版本安装对应 CUDA 版。安装后检查：

```powershell
python --version
python -c "import torch; print('torch =', torch.__version__); print('cuda =', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

期望看到：

```text
cuda = True
显卡名称
```

## 6. 检查公开数据库

在 `Training Code` 目录执行：

```powershell
Test-Path "..\PublicSoilSampleDatabase\public_dataset_manifest.json"
Test-Path "..\PublicSoilSampleDatabase\samples"
(Get-ChildItem "..\PublicSoilSampleDatabase\samples" -Filter *.npz).Count
```

期望样本数量：

```text
4372
```

读取一个样本确认字段：

```powershell
python -c "from pathlib import Path; import numpy as np; root=Path('..')/'PublicSoilSampleDatabase'; p=next((root/'samples').glob('*.npz')); data=np.load(p, allow_pickle=False); print(p.name); print(data['hyper'].shape, data['nir'].shape, data['image'].shape, data['target_names'].tolist(), data['labels']); data.close()"
```

期望结构大致为：

```text
XXXXXXXX.npz
(681,) (5,) (8, 1024, 1024) ['SOC'] [SOC数值]
```

如果数据库在其他位置，用命令临时指定：

```powershell
python Train_main.py --dataset-root "D:\YourPath\PublicSoilSampleDatabase" --dry-run
```

## 7. 查看菜单和模型

查看可选菜单：

```powershell
python Train_main.py --list-menus
```

当前主要菜单：

| 菜单 key | 作用 | 建议 |
| --- | --- | --- |
| `mfpchfnetv2` | MFPC-HFNet 主模型和结构消融。 | 默认先用这个。 |
| `input_ablation` | 输入端消融，例如 NIR-only、Hyper-only、Image+NIR。 | 有 Full 参考结果后再用。 |
| `compare` | ResNet、EfficientNet、ViT、Swin、ConvNeXt、MobileNetV4 等 backbone 对比。 | 计算量大，熟悉后再用。 |

查看某个菜单的模型：

```powershell
python Train_main.py --menu mfpchfnetv2 --list-models
```

默认主模型为：

```text
MFPCHFNetV2_Full
```

## 8. 配置检查和冒烟测试

只检查配置：

```powershell
python Train_main.py --dry-run
```

冒烟测试用于确认环境、数据库、模型构建、Fold 构建、PCA 先验、训练循环和输出目录能跑通。它不是正式结果。

```powershell
python Train_main.py --num-runs 1 --max-epochs 1 --export-onnx-after-training false
```

参数含义：

| 参数 | 作用 |
| --- | --- |
| `--num-runs 1` | 只运行 1 个 Fold。 |
| `--max-epochs 1` | 每个 Fold 只训练 1 个 epoch。 |
| `--export-onnx-after-training false` | 冒烟测试时先不导出 ONNX。 |

## 9. 正式训练

按 `Train_main.py` 当前面板默认值启动：

```powershell
python Train_main.py
```

常用默认值：

```text
TRAIN_MENU = "mfpchfnetv2"
TRAIN_MODEL_NAMES = ["MFPCHFNetV2_Full"]
TARGET_MODE = "soc"
DATASET_ROOT = DEFAULT_PUBLIC_DATABASE_ROOT
NUM_FOLDS = 8
NUM_RUNS = 8
MAX_EPOCHS = 1000
EXPORT_ONNX_AFTER_TRAINING = True
```

只跑前 1 个 Fold，用于较长调试：

```powershell
python Train_main.py --num-runs 1
```

指定输出根目录：

```powershell
python Train_main.py --base-run-dir "D:\MFPC-HFNet-Runs"
```

## 10. 训练输出

训练输出默认写入：

```text
Training Code/ModelData/
```

典型结构：

```text
ModelData/
  2026-xx-xx_xx-xx-xx_mfpchfnetv2_SOC_8FoldCV/
    MFPCHFNetV2_Full/
      Fold01/
        best_model.pth
        metrics_summary.csv
        test_predictions_fold_01_soc.csv
        run_info.json
        pca_priors_train_only.pt
      final_test_metrics_value_pm_std.csv
      test_predictions_all_folds.csv
      ONNX/
```

常用结果文件：

| 文件 | 位置 | 作用 |
| --- | --- | --- |
| `metrics_summary.csv` | Fold 目录 | 当前 Fold 的 Train/Validation/Test 指标摘要，也是 Fold 完成标志。 |
| `test_predictions_fold_XX_soc.csv` | Fold 目录 | 当前 Fold 测试集逐样本预测值。 |
| `run_info.json` | Fold 目录 | 当前 Fold 的设备、数据、模型、PCA 先验和参数追溯。 |
| `pca_priors_train_only.pt` | Fold 目录 | 只用当前 Fold 的 Train 子集构建的 PCA 先验。 |
| `final_test_metrics_value_pm_std.csv` | 单模型目录 | 所有已完成 Fold 的 Test 指标均值和标准差。 |
| `test_predictions_all_folds.csv` | 单模型目录 | 所有已完成 Fold 的测试集预测汇总。 |
| `ONNX/*.onnx` | `ONNX/` | 代表 Fold 导出的部署模型。 |
| `ONNX/*.export_info.json` | `ONNX/` | ONNX 来源 Fold、输入输出接口、参数量和数据库格式追溯。 |

## 11. 断点续训

如果某次训练中断，继续同一实验目录：

```powershell
python Train_main.py --resume-training true --resume-save-dir "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code\ModelData\2026-xx-xx_xx-xx-xx_mfpchfnetv2_SOC_8FoldCV"
```

注意：

| 项目 | 说明 |
| --- | --- |
| `--resume-save-dir` | 必须指向实验目录根部，不是某个 Fold 目录。 |
| 已完成 Fold | 已经存在 `metrics_summary.csv` 的 Fold 会跳过。 |
| 未完成 Fold | 会优先读取 `resume_from_best_state.pth` 和 `training_progress.json`。 |
| 可比性 | 续训时不要随意改 `SPLIT_SEED`、`NUM_FOLDS`、模型清单或数据库。 |

## 12. 当前折分和 PCA 先验规则

交叉验证默认使用稳定样本 ID 构建 8 折：

```text
stable_split_id = sample_id
fold_id = stable_hash(split_seed, stable_split_id) % NUM_FOLDS
```

每个 run 的划分语义：

| 子集 | 规则 |
| --- | --- |
| Test | 当前 run 对应 Fold。 |
| Validation | 相对 Test 偏移 `VALIDATION_FOLD_OFFSET` 的 Fold。 |
| Train | 其余 Fold。 |

MFPC-HFNet 图像分支的 PCA/归一化先验在训练时按 Fold 构建：

```text
FoldXX/pca_priors_train_only.pt
FoldXX/pca_priors_train_only_summary.json
```

该 Fold 先验只由当前 Train 子集估计，Validation/Test 样本不参与通道归一化、结构向量筛选或 PCA 参数估计。

## 13. 常见问题

### 13.1 找不到公开数据库

先检查默认位置：

```powershell
Test-Path "..\PublicSoilSampleDatabase\public_dataset_manifest.json"
Test-Path "..\PublicSoilSampleDatabase\samples"
```

如果数据库在其他位置：

```powershell
python Train_main.py --dataset-root "D:\Datasets\PublicSoilSampleDatabase" --dry-run
```

### 13.2 `TARGET_MODE` 设置后报错

当前公开数据库只发布 SOC 标签，应保持：

```text
TARGET_MODE = "soc"
```

### 13.3 `torch.cuda.is_available()=False`

先检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

常见原因是 PyTorch 版本、CUDA 版本、NVIDIA 驱动或当前虚拟环境不匹配。正式训练要求 CUDA 可用。

### 13.4 `ModuleNotFoundError: No module named 'cv2'`

安装 OpenCV：

```powershell
python -m pip install opencv-python
```

导入名是 `cv2`，安装包名是 `opencv-python`。

### 13.5 CUDA out of memory

优先考虑：

| 方法 | 说明 |
| --- | --- |
| 关闭其他 GPU 程序 | 释放显存。 |
| 临时减少 `NUM_RUNS` | 分批完成折分训练。 |
| 临时选择更小模型 | 用于开发调试。 |
| 更换输出/缓存磁盘 | 避免系统盘空间不足。 |

不要直接在 `Train_core.py` 中硬改显存策略。优先通过菜单和命令行控制实验规模。

### 13.6 PowerShell 路径有空格或中文

路径必须加英文双引号：

```powershell
Set-Location "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code"
```

如果终端中文显示异常，可先设置 UTF-8 输出：

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
```

## 14. 验证命令

语法检查：

```powershell
python -B -m py_compile Train_main.py Train_config.py Train_core.py Train_export_onnx.py Train_optimizer.py Train_support.py Metrics_core.py Menu_MFPCHFNetV2.py Menu_InputAblation.py Menu_Compare_AllBackbones.py Tool\Tool_CheckTrainingPolicy.py Model_MFPCHFNet.py Model_CompareBackbones.py Model_EfficientNet1024Backbones.py Data_LoaderRuntimeAuto.py Data_DiskCacheRegistry.py Data_BuildPcaPriorsFull.py Data_PublicSampleDatabase.py
```

工程边界检查：

```powershell
python Tool\Tool_CheckTrainingPolicy.py
```

期望输出：

```text
V2 training policy check PASSED.
```

训练配置检查：

```powershell
python Train_main.py --dry-run
```

## 15. 发布和使用边界

面向使用者的公开包建议包含：

| 内容 | 说明 |
| --- | --- |
| 源码 `.py` | 当前训练工程源码。 |
| `README.md` | 当前使用说明。 |
| `.gitignore` | 本地输出排除规则。 |
| `ModelAssets/pca_priors_full.pt` | 默认全局 PCA 先验文件。 |
| `PublicSoilSampleDatabase/` 或下载说明 | 当前公开数据库。 |

面向使用者的公开包不建议包含：

| 内容 | 原因 |
| --- | --- |
| `ModelData/` | 训练输出通常体积较大。 |
| `__pycache__/` | Python 临时缓存。 |
| `.venv/` | 本机虚拟环境。 |
| 临时缓存目录 | 可由训练过程重新生成。 |
| 未明确要求的 checkpoint、ONNX、Tensor/engine 文件 | 大体积产物应按发布目标单独选择。 |

## 16. 维护规则

- 菜单只声明训练意图和模型规格，不读取数据、不保存 checkpoint、不导出指标。
- 训练循环只能位于 `Train_core.py`，不得新增多个同级训练主体。
- 模型结构只能位于 `Model_*.py`，菜单通过 `ModelSpec` 和菜单 `Config` 显式传参。
- 数据读取、公开库接口和缓存管理应放在 `Data_*.py`。
- 指标、散点图、CSV/Excel 汇总应放在 `Metrics_core.py` 或同类 `Metrics_` 文件。
- ONNX 选择、dummy input、导出文件命名和追溯 JSON 应放在 `Train_export_onnx.py`。
- Tool 文件只能用于检查、诊断或一次性试验，不应成为训练运行依赖。
- 修改 `Train_main.py` 面板前必须先得到用户确认。

## 17. 关键维护记录

- 2026-06-30：GG 清理 README 中面向使用者不需要的历史说明和数据生成说明，改为只按当前公开数据库结构、训练入口、运行命令和验证命令说明。
