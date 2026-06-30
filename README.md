# MFPC-HFNet 公开版训练工程说明书 / MFPC-HFNet Public Training Project Guide

更新时间：2026-06-30  
English: Last updated: 2026-06-30.  
维护者：ljy / GG  
English: Maintainer: ljy / GG.  
适用范围：公开版 MFPC-HFNet / 多源土壤属性反演训练代码  
English: Scope: public MFPC-HFNet / multisource soil-property inversion training code.  
默认目标：SOC  
English: Default target: SOC.  
默认数据库格式：`public_single_npz_v1`  
English: Default database format: `public_single_npz_v1`.

## 0. 这份说明书怎么用 / How to Use This Guide

这份 README 的目标是让深度学习和编程新手也能从零开始跑通当前公开版训练工程。
English: This README is intended to help deep-learning and programming beginners run the current public training project from scratch.

建议按下面顺序阅读和执行：
English: Read and execute the steps in this order:

1. 先看第 1-4 节，理解工程作用、目录结构、数据库字段和源码职责。  
   English: First read Sections 1-4 to understand the project role, folder layout, database fields, and source-code responsibilities.
2. 再看第 5 节，准备 Python 3.12、CUDA PyTorch 和通用依赖。  
   English: Then read Section 5 to prepare Python 3.12, CUDA PyTorch, and common dependencies.
3. 按第 6 节检查公开数据库数量和字段。  
   English: Follow Section 6 to check the public database count and fields.
4. 按第 7-8 节查看菜单、执行 `--dry-run` 和冒烟测试。  
   English: Follow Sections 7-8 to list menus, run `--dry-run`, and perform a smoke test.
5. 按第 9-12 节启动正式训练、查看结果、断点续训并理解折分和 PCA 先验规则。  
   English: Follow Sections 9-12 to start full training, inspect outputs, resume training, and understand fold and PCA-prior rules.
6. 训练或环境报错时先看第 13 节常见问题。  
   English: If training or environment errors occur, check Section 13 first.

如果只想确认训练入口是否能读取当前配置，最小命令是：
English: If you only want to confirm that the training entry can read the current configuration, use:

```powershell
Set-Location "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code"
python Train_main.py --dry-run
```

`--dry-run` 只打印最终配置，不训练模型，也不会生成正式结果。
English: `--dry-run` only prints the final configuration; it does not train the model or generate formal results.

## 1. 工程作用 / Project Role

本工程用于训练和导出多源土壤属性反演模型。当前公开版围绕 `PublicSoilSampleDatabase` 单文件样本数据库运行，默认训练目标为 SOC，默认主模型为 `MFPCHFNetV2_Full`。
English: This project trains and exports multisource soil-property inversion models. The current public version runs on the single-file sample database `PublicSoilSampleDatabase`, uses SOC as the default target, and uses `MFPCHFNetV2_Full` as the default main model.

当前默认训练任务如下：
English: The current default training task is:

```text
训练菜单 / Menu: mfpchfnetv2
训练模型 / Model: MFPCHFNetV2_Full
目标变量 / Target: SOC
数据库 / Database: PublicSoilSampleDatabase beside Training Code
交叉验证 / Cross-validation: 8 folds, 8 runs
输出目录 / Output directory: Training Code/ModelData
训练后导出 / Export after training: ONNX
```

新手可以把整个工程理解成四件事：
English: Beginners can understand the whole project as four jobs:

| 中文说明<br>EN: Chinese item | 通俗理解<br>EN: Plain explanation | 对应文件<br>EN: Related files |
| --- | --- | --- |
| 读数据<br>EN: Read data | 从公开数据库读取图像、光谱、NIR 和 SOC 标签。<br>EN: Read image, spectrum, NIR, and SOC labels from the public database. | `Data_PublicSampleDatabase.py`, `Data_LoaderRuntimeAuto.py` |
| 分数据<br>EN: Split data | 按稳定样本 ID 构建 8 折交叉验证。<br>EN: Build 8-fold cross-validation from stable sample IDs. | `Train_support.py`, `Train_core.py` |
| 训模型<br>EN: Train models | 按菜单声明训练 MFPC-HFNet 或对比模型。<br>EN: Train MFPC-HFNet or comparison models declared by menus. | `Menu_*.py`, `Model_*.py`, `Train_core.py` |
| 存结果<br>EN: Save results | 保存指标、预测表、模型权重和 ONNX。<br>EN: Save metrics, prediction tables, model weights, and ONNX files. | `Metrics_core.py`, `Train_export_onnx.py`, `ModelData/` |

## 2. 推荐目录结构 / Recommended Directory Structure

默认代码假定 `Training Code` 和 `PublicSoilSampleDatabase` 是并列目录。
English: The default code assumes that `Training Code` and `PublicSoilSampleDatabase` are sibling directories.

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
    Model_CompareBackbones.py
    Model_EfficientNet1024Backbones.py
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
English: Example local paths on this machine:

```text
C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code
C:\Users\94562\Desktop\Multisource Data LJY 2025\PublicSoilSampleDatabase
```

注意事项如下：
English: Important notes:

| 项目<br>EN: Item | 说明<br>EN: Description |
| --- | --- |
| 路径空格<br>EN: Space in path | `Training Code` 路径中有空格，PowerShell 命令里必须加英文双引号。<br>EN: The path `Training Code` contains a space, so PowerShell commands must use double quotes. |
| 默认数据库<br>EN: Default database | 训练入口会自动寻找 `..\PublicSoilSampleDatabase`。<br>EN: The training entry automatically looks for `..\PublicSoilSampleDatabase`. |
| 临时换库<br>EN: Temporary database override | 数据库在其他位置时，优先用 `--dataset-root` 临时覆盖。<br>EN: If the database is stored elsewhere, prefer a temporary `--dataset-root` override. |
| 输出目录<br>EN: Output directory | `ModelData/` 用于保存模型、日志、预测表和汇总表，训练后可能很大。<br>EN: `ModelData/` stores models, logs, prediction tables, and summaries, and can become large after training. |

## 3. 当前公开数据库结构 / Current Public Database Structure

公开数据库格式为：
English: The public database format is:

```text
public_single_npz_v1
```

使用者需要关心的数据库结构如下：
English: The user-facing database structure is:

```text
PublicSoilSampleDatabase/
  public_dataset_manifest.json
  samples/
    8位样本ID.npz
```

每个 `.npz` 样本包含：
English: Each `.npz` sample contains:

| 字段<br>EN: Field | 含义<br>EN: Meaning | 形状或示例<br>EN: Shape or example | 训练用途<br>EN: Training use |
| --- | --- | --- | --- |
| `sample_id` | 样本 ID，与文件名主干一致。<br>EN: Sample ID, same as the file stem. | `00465C9D` | 稳定折分和输出对齐。<br>EN: Stable split and output alignment. |
| `hyper` | HyperVISNIR 光谱特征。<br>EN: HyperVISNIR spectral feature. | `(681,)` | 光谱输入分支。<br>EN: Spectral input branch. |
| `nir` | NIR 数值特征。<br>EN: NIR numeric feature. | `(5,)` | NIR 输入分支。<br>EN: NIR input branch. |
| `image` | 8 通道图像张量。<br>EN: 8-channel image tensor. | `(8, 1024, 1024)` | 图像输入分支。<br>EN: Image input branch. |
| `labels` | 标签数组。<br>EN: Label array. | `(1,)` | SOC 训练目标值。<br>EN: SOC target value. |
| `target_names` | 标签名称数组。<br>EN: Target-name array. | `["SOC"]` | 说明 `labels` 中每个位置的目标变量。<br>EN: Defines target names for positions in `labels`. |

当前公开库只发布 SOC 标签，因此保持：
English: The current public database releases SOC labels only, so keep:

```text
TARGET_MODE = "soc"
```

图像通道波长如下：
English: Image-channel wavelengths:

```text
0490, 0540, 0590, 0660, 0775, 0880, 0945, 1000
```

## 4. 主要源码职责 / Main Source-Code Responsibilities

源码采用“入口、配置、菜单、训练引擎、模型、数据、输出”分层。这样做的好处是：训练循环只在一个地方维护，模型清单由菜单声明，数据读取和结果输出各自独立。
English: The source code is layered into entry, configuration, menus, training engine, models, data, and outputs. This keeps the training loop in one place, model lists in menus, and data/output logic separate.

| 类型<br>EN: Type | 文件或目录<br>EN: File or directory | 作用<br>EN: Purpose |
| --- | --- | --- |
| 训练入口<br>EN: Training entry | `Train_main.py` | 手工参数面板和终端入口。<br>EN: Manual parameter panel and terminal entry point. |
| 参数装配<br>EN: Configuration assembly | `Train_config.py` | 合并菜单默认值、入口面板值和命令行临时覆盖。<br>EN: Merges menu defaults, entry-panel values, and temporary command-line overrides. |
| 训练引擎<br>EN: Training engine | `Train_core.py` | 执行 Dataset、Fold、模型、优化器、训练循环、断点和汇总。<br>EN: Runs Dataset, folds, models, optimizers, training loop, checkpoints, and summaries. |
| 折分/日志工具<br>EN: Split and logging tools | `Train_support.py` | 稳定折分、batch 规划、日志写出等训练支持逻辑。<br>EN: Stable split, batch planning, log writing, and training support. |
| 优化器策略<br>EN: Optimizer policy | `Train_optimizer.py` | 构建 AdamW 参数组、分组学习率和冻结/解冻策略。<br>EN: Builds AdamW parameter groups, grouped learning rates, and freeze/unfreeze policies. |
| 数据读取<br>EN: Data loading | `Data_PublicSampleDatabase.py`, `Data_LoaderRuntimeAuto.py` | 读取公开数据库、按菜单裁剪输入源、组织 Dataset 条目。<br>EN: Reads the public database, trims inputs by menu settings, and builds Dataset items. |
| PCA 先验<br>EN: PCA priors | `Data_BuildPcaPriorsFull.py` | 训练时按当前 Fold 的 Train 子集构建图像分支 PCA/归一化先验。<br>EN: Builds image-branch PCA/normalization priors from the Train subset of each fold. |
| 模型结构<br>EN: Model architecture | `Model_*.py` | 定义 MFPC-HFNet 和对比模型结构。<br>EN: Defines MFPC-HFNet and comparison model architectures. |
| 训练菜单<br>EN: Training menus | `Menu_*.py` | 声明模型清单、输入尺寸、batch size 和模型级策略。<br>EN: Declares model lists, input sizes, batch sizes, and model-level policies. |
| 指标输出<br>EN: Metrics output | `Metrics_core.py` | 保存指标、预测表、散点图和最终汇总。<br>EN: Saves metrics, prediction tables, scatter plots, and final summaries. |
| ONNX 导出<br>EN: ONNX export | `Train_export_onnx.py` | 训练后选择代表 Fold，导出 ONNX 和追溯 JSON。<br>EN: Selects a representative fold after training and exports ONNX plus trace JSON. |
| 工程检查<br>EN: Engineering check | `Tool/Tool_CheckTrainingPolicy.py` | 检查训练工程边界。<br>EN: Checks training-project boundaries. |

参数优先级固定为：
English: Parameter priority is fixed as:

```text
菜单显式设置 > Train_main.py 面板补充设置 > 训练代码默认
Menu explicit settings > Train_main.py panel supplements > training-code defaults
```

命令行参数只作为本次运行的临时覆盖，不会回写源码。
English: Command-line arguments are temporary overrides for the current run only; they do not write back to source code.

## 5. Python 与 CUDA 环境准备 / Python and CUDA Environment Setup

推荐使用 Python 3.12。正式训练默认要求 CUDA GPU。
English: Python 3.12 is recommended. Full training expects a CUDA GPU by default.

### 5.1 创建虚拟环境 / Create a Virtual Environment

进入 `Training Code` 后创建虚拟环境：
English: Enter `Training Code` and create the virtual environment:

```powershell
Set-Location "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 提示禁止运行脚本，可在当前窗口临时放开执行策略：
English: If PowerShell blocks script execution, temporarily relax the execution policy for the current window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 5.2 安装依赖 / Install Dependencies

PyTorch 需要按本机显卡、驱动和 CUDA 版本安装对应 CUDA 版。不要盲目复制其他电脑的 PyTorch 安装命令。
English: Install the CUDA build of PyTorch that matches the local GPU, driver, and CUDA version. Do not blindly copy PyTorch installation commands from another computer.

通用依赖：
English: Common dependencies:

```powershell
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn opencv-python openpyxl onnx thop timm
```

依赖作用说明：
English: Dependency roles:

| 包名<br>EN: Package | 作用<br>EN: Purpose |
| --- | --- |
| `torch` | 深度学习训练主体。<br>EN: Main deep-learning training framework. |
| `torchvision` | 部分 CNN/Transformer backbone 对比模型。<br>EN: Some CNN/Transformer comparison backbones. |
| `numpy` | `.npz` 数据读取和数组计算。<br>EN: `.npz` reading and array computation. |
| `pandas` | CSV/Excel 表格读取与结果汇总。<br>EN: CSV/Excel reading and result summarization. |
| `scipy` | 科学计算和部分数据处理依赖。<br>EN: Scientific computation and selected data-processing support. |
| `opencv-python` | `cv2` 图像读取、滤波和 resize。<br>EN: `cv2` image reading, filtering, and resizing. |
| `scikit-learn` | R^2、RMSE、MAE 等指标。<br>EN: Metrics such as R^2, RMSE, and MAE. |
| `matplotlib` / `seaborn` | 回归散点图输出。<br>EN: Regression scatter-plot output. |
| `openpyxl` | Excel 评价报告写出。<br>EN: Excel report writing. |
| `onnx` | ONNX 导出和检查。<br>EN: ONNX export and checking. |
| `thop` | 导出时估算 FLOPs，失败时不影响主要训练结果。<br>EN: Estimates FLOPs during export; failure does not invalidate core training results. |
| `timm` | MobileNetV4 等部分 backbone 对比模型。<br>EN: Some backbone comparison models such as MobileNetV4. |

### 5.3 检查 Python 和 CUDA / Check Python and CUDA

运行：
English: Run:

```powershell
python --version
python -c "import torch; print('torch =', torch.__version__); print('cuda =', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

期望看到：
English: Expected:

```text
cuda = True
显卡名称 / GPU name
```

如果看到 `cuda = False`，先不要正式训练。常见原因是安装了 CPU 版 PyTorch、NVIDIA 驱动不可用，或当前 Python 环境不是刚安装 CUDA 版 PyTorch 的环境。
English: If `cuda = False`, do not start full training yet. Common causes are a CPU-only PyTorch installation, unavailable NVIDIA driver, or using a different Python environment from the one with CUDA PyTorch installed.

## 6. 检查公开数据库 / Check the Public Database

### 6.1 默认数据库位置 / Default Database Location

公开数据库格式为 `public_single_npz_v1`，默认数据库根目录为 `Training Code` 上一级的 `PublicSoilSampleDatabase`。
English: The public database format is `public_single_npz_v1`, and the default database root is `PublicSoilSampleDatabase` beside `Training Code`.

在 `Training Code` 目录执行：
English: Run in the `Training Code` directory:

```powershell
Test-Path "..\PublicSoilSampleDatabase\public_dataset_manifest.json"
Test-Path "..\PublicSoilSampleDatabase\samples"
(Get-ChildItem "..\PublicSoilSampleDatabase\samples" -Filter *.npz).Count
```

期望样本数量：
English: Expected sample count:

```text
4372
```

### 6.2 读取一个样本 / Read One Sample

这个命令只读取一个 `.npz`，用于确认 Python 能打开数据库并看到正确字段。
English: This command reads only one `.npz` file to confirm that Python can open the database and see the expected fields.

```powershell
python -c "from pathlib import Path; import numpy as np; root=Path('..')/'PublicSoilSampleDatabase'; p=next((root/'samples').glob('*.npz')); data=np.load(p, allow_pickle=False); print(p.name); print(data['hyper'].shape, data['nir'].shape, data['image'].shape, data['target_names'].tolist(), data['labels']); data.close()"
```

期望结构大致为：
English: Expected structure:

```text
XXXXXXXX.npz
(681,) (5,) (8, 1024, 1024) ['SOC'] [SOC数值 / SOC value]
```

### 6.3 临时指定数据库路径 / Temporarily Specify Database Path

如果数据库在其他位置，不建议直接改源码。优先使用命令行临时覆盖：
English: If the database is stored elsewhere, do not edit source code first. Prefer a temporary command-line override:

```powershell
python Train_main.py --dataset-root "D:\YourPath\PublicSoilSampleDatabase" --dry-run
```

## 7. 查看菜单和模型 / List Menus and Models

查看可选菜单：
English: List available menus:

```powershell
python Train_main.py --list-menus
```

当前主要菜单如下：
English: Current main menus:

| 菜单 key<br>EN: Menu key | 作用<br>EN: Purpose | 新手建议<br>EN: Beginner suggestion |
| --- | --- | --- |
| `mfpchfnetv2` | MFPC-HFNet 主模型和结构消融。<br>EN: MFPC-HFNet main model and architecture ablation. | 默认先用这个。<br>EN: Use this first by default. |
| `input_ablation` | 输入端消融，例如 NIR-only、Hyper-only、Image+NIR。<br>EN: Input-side ablation, such as NIR-only, Hyper-only, and Image+NIR. | 有 Full 参考结果后再用。<br>EN: Use after obtaining a Full reference result. |
| `compare` | ResNet、EfficientNet、ViT、Swin、ConvNeXt、MobileNetV4 等 backbone 对比。<br>EN: Backbone comparison such as ResNet, EfficientNet, ViT, Swin, ConvNeXt, and MobileNetV4. | 计算量大，熟悉后再用。<br>EN: Computationally heavy; use after becoming familiar with the project. |

查看某个菜单有哪些模型：
English: List models under a menu:

```powershell
python Train_main.py --menu mfpchfnetv2 --list-models
```

默认主模型为：
English: The default main model is:

```text
MFPCHFNetV2_Full
```

## 8. 第一次运行：配置检查和冒烟测试 / First Run: Configuration Check and Smoke Test

### 8.1 只检查配置 / Configuration Check Only

运行：
English: Run:

```powershell
python Train_main.py --dry-run
```

重点检查以下项目：
English: Focus on the following items:

| 项目<br>EN: Item | 应看到的内容<br>EN: Expected content |
| --- | --- |
| `菜单接口` | `mfpchfnetv2` |
| `训练模型` | `MFPCHFNetV2_Full` |
| `TARGET_MODE` | `soc` |
| `DATASET_ROOT` | 指向当前 `PublicSoilSampleDatabase`。<br>EN: Points to the current `PublicSoilSampleDatabase`. |
| `DATA_DIR` | 指向当前 `PublicSoilSampleDatabase\samples`。<br>EN: Points to the current `PublicSoilSampleDatabase\samples`. |
| `NUM_FOLDS` / `NUM_RUNS` | `8` / `8` |

### 8.2 冒烟测试 / Smoke Test

冒烟测试用于确认环境、数据库、模型构建、Fold 构建、PCA 先验、训练循环和输出目录都能跑通。它不是正式结果，不能用于论文或部署。
English: The smoke test confirms that environment, database, model construction, fold construction, PCA priors, training loop, and output directory work. It is not a formal result and should not be used for papers or deployment.

```powershell
python Train_main.py --num-runs 1 --max-epochs 1 --export-onnx-after-training false
```

参数含义：
English: Parameter meanings:

| 参数<br>EN: Parameter | 作用<br>EN: Purpose |
| --- | --- |
| `--num-runs 1` | 只运行 1 个 Fold。<br>EN: Runs only one fold. |
| `--max-epochs 1` | 每个 Fold 只训练 1 个 epoch。<br>EN: Trains only one epoch per fold. |
| `--export-onnx-after-training false` | 冒烟测试时先不导出 ONNX。<br>EN: Disables ONNX export during the smoke test. |

## 9. 正式训练 / Full Training

按 `Train_main.py` 当前面板默认值启动：
English: Start training with the current default panel values in `Train_main.py`:

```powershell
python Train_main.py
```

常用默认值如下：
English: Common default values:

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
English: Run only the first fold for longer debugging:

```powershell
python Train_main.py --num-runs 1
```

指定输出根目录：
English: Specify an output root:

```powershell
python Train_main.py --base-run-dir "D:\MFPC-HFNet-Runs"
```

## 10. 训练输出 / Training Outputs

训练输出默认写入：
English: Training outputs are written by default to:

```text
Training Code/ModelData/
```

典型结构如下：
English: Typical structure:

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
        pca_priors_train_only_summary.json
      final_test_metrics_value_pm_std.csv
      test_predictions_all_folds.csv
      ONNX/
        DLV-3-Both-Full.onnx
        DLV-3-Both-Full.export_info.json
```

常用结果文件如下：
English: Common output files:

| 文件<br>EN: File | 位置<br>EN: Location | 作用<br>EN: Purpose |
| --- | --- | --- |
| `metrics_summary.csv` | Fold 目录<br>EN: Fold directory | 当前 Fold 的 Train/Validation/Test 指标摘要，也是 Fold 完成标志。<br>EN: Train/Validation/Test metric summary for the fold, also the fold-completion marker. |
| `test_predictions_fold_XX_soc.csv` | Fold 目录<br>EN: Fold directory | 当前 Fold 测试集逐样本预测值。<br>EN: Per-sample predictions for the test set of the fold. |
| `run_info.json` | Fold 目录<br>EN: Fold directory | 当前 Fold 的设备、数据、模型、PCA 先验和参数追溯。<br>EN: Device, data, model, PCA-prior, and parameter trace for the fold. |
| `pca_priors_train_only.pt` | Fold 目录<br>EN: Fold directory | 只用当前 Fold 的 Train 子集构建的 PCA 先验。<br>EN: PCA priors built only from the Train subset of the current fold. |
| `pca_priors_train_only_summary.json` | Fold 目录<br>EN: Fold directory | PCA 先验构建摘要。<br>EN: Summary of PCA-prior construction. |
| `final_test_metrics_value_pm_std.csv` | 单模型目录<br>EN: Single-model directory | 所有已完成 Fold 的 Test 指标均值和标准差。<br>EN: Mean and standard deviation of Test metrics across completed folds. |
| `test_predictions_all_folds.csv` | 单模型目录<br>EN: Single-model directory | 所有已完成 Fold 的测试集预测汇总。<br>EN: Aggregated test predictions across completed folds. |
| `ONNX/*.onnx` | `ONNX/` | 代表 Fold 导出的部署模型。<br>EN: Deployment model exported from the representative fold. |
| `ONNX/*.export_info.json` | `ONNX/` | ONNX 来源 Fold、输入输出接口、参数量和数据库格式追溯。<br>EN: Trace of ONNX source fold, input/output interface, parameter count, and database format. |

## 11. 断点续训 / Resume Training

如果某次训练中断，可继续同一实验目录：
English: If training is interrupted, continue the same experiment directory:

```powershell
python Train_main.py --resume-training true --resume-save-dir "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code\ModelData\2026-xx-xx_xx-xx-xx_mfpchfnetv2_SOC_8FoldCV"
```

注意事项：
English: Notes:

| 项目<br>EN: Item | 说明<br>EN: Description |
| --- | --- |
| `--resume-save-dir` | 必须指向实验目录根部，不是某个 Fold 目录。<br>EN: Must point to the experiment root, not to a single fold directory. |
| 已完成 Fold<br>EN: Completed fold | 已经存在 `metrics_summary.csv` 的 Fold 会跳过。<br>EN: A fold with `metrics_summary.csv` is skipped. |
| 未完成 Fold<br>EN: Unfinished fold | 会优先读取 `resume_from_best_state.pth` 和 `training_progress.json`。<br>EN: Reads `resume_from_best_state.pth` and `training_progress.json` first. |
| 可比性<br>EN: Comparability | 续训时不要随意改 `SPLIT_SEED`、`NUM_FOLDS`、模型清单或数据库。<br>EN: Do not casually change `SPLIT_SEED`, `NUM_FOLDS`, model list, or database during resume. |

## 12. 当前折分和 PCA 先验规则 / Current Fold and PCA-Prior Rules

### 12.1 稳定折分 / Stable Fold Assignment

交叉验证默认使用稳定样本 ID 构建 8 折。
English: Cross-validation builds 8 folds from stable sample IDs by default.

普通文本公式如下：
English: Plain-text formula:

```text
stable_split_id = sample_id

fold_id = stable_hash(split_seed, stable_split_id) % NUM_FOLDS
```

每个 run 的划分语义：
English: Split meaning for each run:

| 子集<br>EN: Subset | 规则<br>EN: Rule |
| --- | --- |
| Test | 当前 run 对应 Fold。<br>EN: Fold corresponding to the current run. |
| Validation | 相对 Test 偏移 `VALIDATION_FOLD_OFFSET` 的 Fold。<br>EN: Fold offset from Test by `VALIDATION_FOLD_OFFSET`. |
| Train | 其余 Fold。<br>EN: All remaining folds. |

### 12.2 Fold 内 PCA 先验 / Fold-Local PCA Priors

MFPC-HFNet 图像分支的 PCA/归一化先验在训练时按 Fold 构建。
English: The PCA/normalization priors for the MFPC-HFNet image branch are built per fold during training.

输出文件：
English: Output files:

```text
FoldXX/pca_priors_train_only.pt
FoldXX/pca_priors_train_only_summary.json
```

该 Fold 先验只由当前 Train 子集估计，Validation/Test 样本不参与通道归一化、结构向量筛选或 PCA 参数估计。
English: The fold priors are estimated only from the current Train subset. Validation/Test samples do not participate in channel normalization, structural-vector screening, or PCA parameter estimation.

## 13. 常见问题 / Troubleshooting

### 13.1 找不到公开数据库 / Cannot Find the Public Database

先检查默认位置：
English: First check the default location:

```powershell
Test-Path "..\PublicSoilSampleDatabase\public_dataset_manifest.json"
Test-Path "..\PublicSoilSampleDatabase\samples"
```

如果数据库在其他位置：
English: If the database is elsewhere:

```powershell
python Train_main.py --dataset-root "D:\Datasets\PublicSoilSampleDatabase" --dry-run
```

### 13.2 `TARGET_MODE` 设置后报错 / Error After Setting `TARGET_MODE`

当前公开数据库只发布 SOC 标签，应保持：
English: The current public database releases SOC labels only, so keep:

```text
TARGET_MODE = "soc"
```

### 13.3 `torch.cuda.is_available()=False`

先检查：
English: First check:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

常见原因是 PyTorch 版本、CUDA 版本、NVIDIA 驱动或当前虚拟环境不匹配。正式训练要求 CUDA 可用。
English: Common causes are mismatch among PyTorch version, CUDA version, NVIDIA driver, or the active virtual environment. Full training requires CUDA.

### 13.4 `ModuleNotFoundError: No module named 'cv2'`

安装 OpenCV：
English: Install OpenCV:

```powershell
python -m pip install opencv-python
```

导入名是 `cv2`，安装包名是 `opencv-python`。
English: The import name is `cv2`, while the package name is `opencv-python`.

### 13.5 CUDA out of memory

优先考虑：
English: Prefer these actions:

| 方法<br>EN: Method | 说明<br>EN: Description |
| --- | --- |
| 关闭其他 GPU 程序<br>EN: Close other GPU programs | 释放显存。<br>EN: Frees GPU memory. |
| 临时减少 `NUM_RUNS`<br>EN: Temporarily reduce `NUM_RUNS` | 分批完成折分训练。<br>EN: Completes fold training in smaller batches. |
| 临时选择更小模型<br>EN: Temporarily choose a smaller model | 用于开发调试。<br>EN: Useful for development debugging. |
| 更换输出/缓存磁盘<br>EN: Move output/cache disk | 避免系统盘空间不足。<br>EN: Avoids running out of system-disk space. |

不要直接在 `Train_core.py` 中硬改显存策略。优先通过菜单和命令行控制实验规模。
English: Do not directly hard-code memory strategy changes in `Train_core.py`. Prefer controlling experiment scale through menus and command-line options.

### 13.6 PowerShell 路径有空格或中文 / Spaces or Chinese Characters in PowerShell Paths

路径必须加英文双引号：
English: Paths must be wrapped in English double quotes:

```powershell
Set-Location "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code"
```

如果终端中文显示异常，可先设置 UTF-8 输出：
English: If Chinese text displays incorrectly in the terminal, set UTF-8 output first:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
```

## 14. 验证命令 / Verification Commands

语法检查：
English: Syntax check:

```powershell
python -B -m py_compile Train_main.py Train_config.py Train_core.py Train_export_onnx.py Train_optimizer.py Train_support.py Metrics_core.py Menu_MFPCHFNetV2.py Menu_InputAblation.py Menu_Compare_AllBackbones.py Tool\Tool_CheckTrainingPolicy.py Model_MFPCHFNet.py Model_CompareBackbones.py Model_EfficientNet1024Backbones.py Data_LoaderRuntimeAuto.py Data_DiskCacheRegistry.py Data_BuildPcaPriorsFull.py Data_PublicSampleDatabase.py
```

工程边界检查：
English: Engineering-boundary check:

```powershell
python Tool\Tool_CheckTrainingPolicy.py
```

期望输出：
English: Expected output:

```text
V2 training policy check PASSED.
```

训练配置检查：
English: Training-configuration check:

```powershell
python Train_main.py --dry-run
```

## 15. 发布和使用边界 / Release and Usage Boundary

### 15.1 数据获取与体积 / Data Access and Size

当前随包公开数据库 `PublicSoilSampleDatabase` 的本地实测体积如下：
English: The measured local size of the packaged public database `PublicSoilSampleDatabase` is:

```text
146,718,730,708 bytes
约 136.64 GiB
约 146.72 GB
```

如需索要原数据或完整数据获取方式，请向以下邮箱申请：
English: To request the original data or access instructions for the complete data, please contact:

```text
foddcus@gmail
```

申请时建议说明数据用途、使用单位、联系人和预期使用范围。
English: When requesting data, it is recommended to include the intended use, organization, contact person, and expected usage scope.

### 15.2 公开包建议内容 / Recommended Public Package Contents

面向使用者的公开包建议包含：
English: The user-facing public package should include:

| 内容<br>EN: Content | 说明<br>EN: Description |
| --- | --- |
| 源码 `.py`<br>EN: Source `.py` files | 当前训练工程源码。<br>EN: Current training-project source code. |
| `README.md` | 当前使用说明。<br>EN: Current usage guide. |
| `.gitignore` | 本地输出排除规则。<br>EN: Local-output exclusion rules. |
| `ModelAssets/pca_priors_full.pt` | 默认全局 PCA 先验文件。<br>EN: Default global PCA-prior file. |
| `PublicSoilSampleDatabase/` 或下载说明<br>EN: `PublicSoilSampleDatabase/` or download note | 当前公开数据库。<br>EN: Current public database. |

面向使用者的公开包不建议包含：
English: The user-facing public package should not include:

| 内容<br>EN: Content | 原因<br>EN: Reason |
| --- | --- |
| `ModelData/` | 训练输出通常体积较大。<br>EN: Training outputs are usually large. |
| `__pycache__/` | Python 临时缓存。<br>EN: Python temporary cache. |
| `.venv/` | 本机虚拟环境。<br>EN: Local virtual environment. |
| 临时缓存目录<br>EN: Temporary cache directories | 可由训练过程重新生成。<br>EN: Can be regenerated by training. |
| 未明确要求的 checkpoint、ONNX、Tensor/engine 文件<br>EN: Unrequested checkpoint, ONNX, Tensor/engine files | 大体积产物应按发布目标单独选择。<br>EN: Large artifacts should be selected separately according to the release goal. |

## 16. 维护规则 / Maintenance Rules

- 菜单只声明训练意图和模型规格，不读取数据、不保存 checkpoint、不导出指标。  
  English: Menus only declare training intent and model specifications; they do not read data, save checkpoints, or export metrics.
- 训练循环只能位于 `Train_core.py`，不得新增多个同级训练主体。  
  English: The training loop must remain in `Train_core.py`; do not add multiple peer training bodies.
- 模型结构只能位于 `Model_*.py`，菜单通过 `ModelSpec` 和菜单 `Config` 显式传参。  
  English: Model architectures must remain in `Model_*.py`; menus pass parameters explicitly through `ModelSpec` and menu `Config`.
- 数据读取、公开库接口和缓存管理应放在 `Data_*.py`。  
  English: Data loading, public-database interface, and cache management should stay in `Data_*.py`.
- 指标、散点图、CSV/Excel 汇总应放在 `Metrics_core.py` 或同类 `Metrics_` 文件。  
  English: Metrics, scatter plots, and CSV/Excel summaries should stay in `Metrics_core.py` or similar `Metrics_` files.
- ONNX 选择、dummy input、导出文件命名和追溯 JSON 应放在 `Train_export_onnx.py`。  
  English: ONNX selection, dummy inputs, export naming, and trace JSON should stay in `Train_export_onnx.py`.
- Tool 文件只能用于检查、诊断或一次性试验，不应成为训练运行依赖。  
  English: Tool files should be used only for checks, diagnostics, or one-time experiments; they should not become training runtime dependencies.
- 修改 `Train_main.py` 面板前必须先得到用户确认。  
  English: Confirm with the user before modifying the `Train_main.py` panel.

## 17. 关键维护记录 / Key Maintenance Log

- 2026-06-30：GG 新增数据索要邮箱和当前公开数据库体积说明。
  English: 2026-06-30: GG added the data request email and the current public database size note.
- 2026-06-30：GG 补正文档细节和中英文对照，保留当前公开数据库结构、训练入口、运行命令、输出解读和维护边界。
  English: 2026-06-30: GG restored detailed documentation and bilingual notes while keeping the focus on the current public database structure, training entry, run commands, output interpretation, and maintenance boundary.
