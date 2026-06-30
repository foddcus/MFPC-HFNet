# Python script V2 公开版训练工程新手说明书 / Python script V2 public-release training projectbeginnernote.

更新时间：2026-06-18  
English: last updated: 2026-06-18.
维护者：ljy / GG  
English: maintainer: ljy / GG.
适用范围：公开版 MFPC-HFNet / 多源土壤属性反演训练代码  
English: scope: public release MFPC-HFNet / more source soil soil property property inverse inversion training code.
默认目标：SOC  
English: default target: SOC.

## 0. 这份说明书怎么用 / note use.

这份 README 的目标是让深度学习和编程新手也能从零开始跑通公开版训练工程。建议按下面顺序阅读和执行：
English: README is let deep degree learning learning and programming program beginner also can from zero start start pass public-release training project.Read and execute the steps in the following order:

1. 先看第 1-4 节，确认工程是什么、文件夹放在哪里、电脑环境是否满足。
English: first section 1-4 section, confirm what the project does, where the folders are located, whether the computer environment meets the requirements.
2. 再按第 5 节做数据检查，确认公开数据库能被代码找到。
English: then by section 5 section run the data check, confirm confirm the public database can be found by the code.
3. 按第 6 节先执行 `--dry-run`，只检查配置，不训练。
English: by section 6 section first execute `--dry-run`, only checks the configuration, does not train.
4. 按第 7 节做 1 个 Fold、1 个 epoch 的冒烟测试，确认环境和数据链路能跑。
English: by section 7 section do 1 Fold, 1 epoch smoke test, confirm confirm environment and data pipeline can.
5. 最后按第 8 节启动正式训练。
English: most after by section 8 section start formal training.
6. 训练完成后按第 9 节查看结果文件。
English: traincomplete after by section 9 section inspect the result files.

如果你只想先确认代码是否能启动，最小命令是：
English: result only want first confirm confirm code is no can start, minimum is:

```powershell
Set-Location "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code"
python Train_main.py --dry-run
```

`--dry-run` 只打印最终配置，不会训练模型，也不会生成正式结果。
English: `--dry-run` only print most, does not train the model, also does not generate formal results.

## 1. 工程到底做什么 / program to do.

本工程用于训练和导出多源土壤属性反演模型。当前公开版围绕单文件样本数据库运行，默认训练目标为 SOC，默认主模型为 MFPC-HFNet Full。训练代码不再依赖外部 `.mat` 真值表；样本标签已经写入每个 `.npz` 样本文件。
English: this projectis used to train and export multisource soil-property inversion models.currentpublic releaseruns around the single-file sample database, defaulttrain as SOC, default as MFPC-HFNet Full.training codeno longer depends on external `.mat` value table; sample labels have already been written into each `.npz` sample file.

当前默认训练任务是：
English: current default training task is:

```text
训练菜单：mfpchfnetv2
训练模型：MFPCHFNetV2_Full
目标变量：SOC
数据库：Training Code 上一级的 PublicSoilSampleDatabase
交叉验证：8 Fold，运行 8 个 Fold
输出目录：Training Code/ModelData
训练后导出：ONNX
```

新手可以把整个工程理解成下面几类文件：
English: Beginners can understand the project as the following groups of files:

| 类型<br>EN: type. | 文件前缀 / 文件夹<br>EN: file before / folder. | 通俗理解<br>EN: pass. | 主要作用<br>EN: need purpose. |
| --- | --- | --- | --- |
| 训练菜单<br>EN: training menu. | `Menu_*.py` | 菜单<br>EN: single. | 声明本次要训练哪些模型、顺序、输入尺寸、batch size 和模型级微调策略。<br>EN: need train, order order, input size, batch size and model-level fine-tuning policy. |
| 训练主管<br>EN: training controller. | `Train_main.py` | 主管<br>EN: English counterpart for the Chinese technical note above. | 手工参数面板和主入口，负责把菜单、数据路径、输出路径等交给训练系统。<br>EN: manual parameter panel and main entry point, single, datapath, path train. |
| 配置装配<br>EN: configuration assembly. | `Train_config.py` | 参数整理员<br>EN: parameters. | 合并菜单默认值、`Train_main.py` 面板值和命令行临时覆盖。<br>EN: merges menu defaults, `Train_main.py` value and line temporary override. |
| 训练引擎<br>EN: training engine. | `Train_core.py` | 厨师<br>EN: English counterpart for the Chinese technical note above. | 真正执行 Dataset、Fold、模型、优化器、训练循环、断点和汇总。<br>EN: actuallyexecute Dataset, Fold,,, train environment, and. |
| 优化器策略<br>EN: optimizer policy. | `Train_optimizer.py` | 调参执行员<br>EN: execute. | 根据菜单声明构建 AdamW 参数组、分组学习率和冻结/解冻策略。<br>EN: data single build AdamW parameters, grouped learning rates and freeze/unfreeze policy. |
| 模型结构<br>EN: model architecture. | `Model_*.py` | 食材<br>EN: English counterpart for the Chinese technical note above. | 定义 MFPC-HFNet、baseline/backbone 结构。<br>EN: MFPC-HFNet, baseline/backbone result. |
| 数据读取<br>EN: data loading. | `Data_*.py` | 调料 / 数据库接口<br>EN: / data database interface. | 读取公开库、旧库兼容、blank 校正、缓存、PCA 先验构建。<br>EN: read start database, legacy databasecompatible, blank, cache, PCA first build. |
| 结果输出<br>EN: result output. | `Metrics_core.py` | 上菜员<br>EN: on. | 保存指标、预测表、散点图和最终汇总。<br>EN: savemetrics, prediction table, scatter plot and most. |
| ONNX 导出<br>EN: ONNX export. | `Train_export_onnx.py` | 部署导出员<br>EN: export. | 训练后选择代表 Fold，导出应用端 ONNX 和追溯 JSON。<br>EN: train after select table Fold, export should use ONNX and JSON. |
| 检查工具<br>EN: checking tool. | `Tool/` | 体检工具<br>EN: English counterpart for the Chinese technical note above. | 检查训练工程边界，不作为训练入口。<br>EN: checktraining project, not as train interface. |

参数优先级固定为：
English: parameter priorityfixed as:

```text
菜单显式设置 > Train_main.py 面板补充设置 > Train_core/CommonTrainConfig 默认
```

命令行参数只作为本次运行的临时覆盖，不会回写源码。
English: command-line argumentsonly acts as a temporary override for this run, does not write back to source code.

## 2. 文件夹应该放成什么样 / folder should this.

默认代码假定 `Training Code` 和 `PublicSoilSampleDatabase` 是并列目录：
English: the default code assumes `Training Code` and `PublicSoilSampleDatabase` are sibling directories:

```text
Multisource Data LJY 2025/
  Training Code/
    README.md
    Train_main.py
    Train_config.py
    Train_core.py
    Menu_MFPCHFNetV2.py
    Model_MFPCHFNet.py
    ModelAssets/
      pca_priors_full.pt
    ModelData/
      训练输出会写到这里
  PublicSoilSampleDatabase/
    README.md
    public_dataset_manifest.json
    rebuild_report.json
    samples/
      00465C9D.npz
      ...
```

本机当前路径示例：
English: currentpath:

```text
C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code
C:\Users\94562\Desktop\Multisource Data LJY 2025\PublicSoilSampleDatabase
```

注意事项：
English: notes:

- `Training Code` 路径中有空格，PowerShell 命令里必须加英文双引号。
English: `Training Code` the path contains spaces, PowerShell must English double quotes.
- 默认训练会自动寻找 `..\PublicSoilSampleDatabase`，即 `Training Code` 上一级的公开数据库。
English: default training automatically searches for `..\PublicSoilSampleDatabase`, `Training Code` on start data database.
- 如果公开数据库放在别的位置，不要先改代码，优先用 `--dataset-root` 临时覆盖。
English: if the public database is placed elsewhere, do not change the code first, prefer using `--dataset-root` temporary override.
- `ModelData/` 是训练输出目录，可能很大，不建议放到网盘同步冲突频繁的位置。
English: `ModelData/` is the training output directory, can can large, is not recommended for locations with frequent cloud-sync conflicts.

## 3. 新手需要先理解的几个词 / beginner need need first.

| 词<br>EN: English counterpart for the Chinese technical note above. | 含义<br>EN: Meaning: | 本工程中的表现<br>EN: this project in table. |
| --- | --- | --- |
| 样本<br>EN: sample. | 一条土壤样本数据。<br>EN: one soil sample record. | 一个 `.npz` 文件，包含 `image`、`hyper`、`nir`、`labels`。<br>EN: `.npz` file, `image`, `hyper`, `nir`, `labels`. |
| 模态<br>EN: modality. | 一类输入数据。<br>EN: one type of input data. | `image` 是 8 通道图像，`hyper` 是 681 维 HyperVISNIR，`nir` 是 5 维 NIR。<br>EN: `image` is 8 pass image, `hyper` is 681 HyperVISNIR, `nir` is 5 NIR. |
| 标签<br>EN: label. | 要预测的真实值。<br>EN: the ground-truth value to predict. | 当前公开版只有 SOC，单位 g/kg。<br>EN: currentpublic releaseonly SOC, unit g/kg. |
| 模型<br>EN: English counterpart for the Chinese technical note above. | 神经网络结构。<br>EN: result. | 默认是 `MFPCHFNetV2_Full`。<br>EN: default is `MFPCHFNetV2_Full`. |
| epoch | 模型完整看一遍训练集。<br>EN: the model sees the whole training set once. | `MAX_EPOCHS=1000` 表示最多训练 1000 轮。<br>EN: `MAX_EPOCHS=1000` means most more train 1000. |
| batch size | 一次送进模型的样本数。<br>EN: the number of samples sent into the model at once. | 菜单声明默认 batch，正式训练前还会做 CUDA 显存预检并可能自动降低。<br>EN: single default batch, formal training before will do CUDA and can can automatically low. |
| Fold | 交叉验证中的一份数据划分。<br>EN: one data partition in cross-validation. | 8 Fold 表示数据被分成 8 份，轮流做 Test。<br>EN: 8 Fold meansdata 8, do Test. |
| run | 运行第几个 Fold。<br>EN: run section Fold. | `NUM_RUNS=8` 表示 8 个 Fold 都跑；调试时可临时设为 1。<br>EN: `NUM_RUNS=8` means 8 Fold all; when can temporary as 1. |
| Validation | 训练过程中用于选最佳模型的验证集。<br>EN: the validation set used to select the best model during training. | 当前 run 的 Test Fold 后偏移 1 个 Fold。<br>EN: current run Test Fold after 1 Fold. |
| Test | 最终评估的测试集。<br>EN: the test set used for final evaluation. | 当前 run 对应的 Fold。<br>EN: current run for should Fold. |
| checkpoint | 训练中保存的权重和进度。<br>EN: weights and progress saved during training. | `best_model.pth`、`resume_from_best_state.pth`、`training_progress.json`。 |
| ONNX | 部署用模型格式。<br>EN: use form. | 训练完成后写入模型目录下的 `ONNX/` 文件夹。<br>EN: traincomplete after write below `ONNX/` folder. |

## 4. 电脑和 Python 环境准备 / computer computer and Python environment environment.

### 4.1 硬件要求 / 4.1 need.

正式训练默认要求 CUDA GPU。代码会主动检查：
English: formal trainingdefault need CUDA GPU. code will check:

```text
torch.cuda.is_available() == True
```

如果不是 CUDA 环境，正式训练会停止，而不是静默退回 CPU 慢跑。这是有意设计，避免新手误以为训练正常但实际在 CPU 上跑很久。
English: result not is CUDA environment environment, formal training will, not is CPU. is, avoidbeginner as train in CPU on.

建议准备：
English: recommended:

- NVIDIA GPU。
- 已安装可用的 NVIDIA 驱动。
English: already can use NVIDIA.
- 与 PyTorch 匹配的 CUDA 版 PyTorch。
English: and PyTorch CUDA PyTorch.
- 足够磁盘空间。当前公开数据库约 136.64 GiB，训练输出还会继续占用空间。
English: meet.current start data database 136.64 GiB, train will use.

### 4.2 创建 Python 虚拟环境 / 4.2 Python environment environment.

推荐使用 Python 3.12。进入 `Training Code` 后创建虚拟环境：
English: use Python 3.12. `Training Code` after environment environment:

```powershell
Set-Location "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 提示禁止运行脚本，可只在当前窗口临时放开执行策略：
English: result PowerShell reportforbiddenrun, can only in current interface temporary start execute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4.3 安装依赖 / 4.3.

PyTorch 必须安装 CUDA 版。由于不同显卡、驱动和 CUDA 版本对应的安装命令不同，新手不要盲目复制别人电脑上的 PyTorch 安装命令。建议先按本机 CUDA 情况安装 `torch` 和 `torchvision`，再装通用依赖。
English: PyTorch must CUDA. by not same, and CUDA for should not same, beginnerdo not computer computer on PyTorch.recommended first by CUDA `torch` and `torchvision`, then general.

通用依赖：
English: general:

```powershell
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn opencv-python openpyxl onnx thop timm
```

依赖作用说明：
English: purposenote:

| 包名<br>EN: name. | 作用<br>EN: Purpose: |
| --- | --- |
| `torch` | 深度学习训练主体。<br>EN: deep degree learning learning train. |
| `torchvision` | baseline/backbone 对比模型和部分 CNN/Transformer 主干。<br>EN: baseline/backbone for and CNN/Transformer stem. |
| `numpy` | `.npz` 数据读取和数组计算。<br>EN: `.npz` data loading and number. |
| `pandas` | CSV/Excel 表格读取与结果汇总。<br>EN: CSV/Excel table read and result summary. |
| `scipy` | 旧库兼容、统计和 `.mat` 兼容读取。<br>EN: legacy databasecompatible, and `.mat` compatibleread. |
| `opencv-python` | `cv2`，图像读取、滤波、resize。<br>EN: `cv2`, imageread,, resize. |
| `scikit-learn` | R^2、RMSE、MAE 等指标。<br>EN: R^2, RMSE, MAE metrics. |
| `matplotlib` / `seaborn` | 回归散点图输出。<br>EN: scatter plot. |
| `openpyxl` | Excel 评价报告写出。<br>EN: Excel write. |
| `onnx` | ONNX 导出和检查相关依赖。<br>EN: ONNX export and check. |
| `thop` | 导出时估算 FLOPs，失败时不影响主要训练结果。<br>EN: export when FLOPs, when does not affect need train result result. |
| `timm` | MobileNetV4 等部分 backbone 对比模型。<br>EN: MobileNetV4 backbone for. |

### 4.4 检查 Python 和 CUDA / 4.4 check Python and CUDA.

```powershell
python --version
python -c "import torch; print('torch =', torch.__version__); print('cuda =', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

期望看到：
English: to:

```text
cuda = True
显卡名称
```

如果看到 `cuda = False`，先不要正式训练。常见原因是安装了 CPU 版 PyTorch、NVIDIA 驱动不可用，或当前 Python 环境不是你刚安装 CUDA 版 PyTorch 的环境。
English: result to `cuda = False`, first do notformal training. is CPU PyTorch, NVIDIA not can use, or current Python environment environment not is CUDA PyTorch environment environment.

## 5. 检查公开数据库 / check start data database.

### 5.1 默认数据库格式 / 5.1 defaultdatabase format.

公开数据库格式为 `public_single_npz_v1`。默认数据库根目录为 `Training Code` 上一级的 `PublicSoilSampleDatabase`。
English: start database format as `public_single_npz_v1`.defaultdata database root directory as `Training Code` on `PublicSoilSampleDatabase`.

公开数据库结构：
English: start data database result:

```text
PublicSoilSampleDatabase/
  public_dataset_manifest.json
  rebuild_report.json
  samples/
    8位样本ID.npz
```

每个 `.npz` 样本文件包含：
English: each `.npz` sample file:

```text
sample_id
hyper
nir
image
labels
target_names
```

当前公开版约束：
English: currentpublic release:

- 一个样本对应一个 `.npz` 文件。
English: sample for should `.npz` file.
- `sample_id` 为 8 位大写字母/数字随机 ID。
English: `sample_id` as 8 large write / number ID.
- 当前公开库只发布 SOC 标签，因此 `TARGET_MODE` 默认并建议保持为 `soc`。
English: current start database only SOC label, therefore `TARGET_MODE` default and recommended as `soc`.
- 如果设置为 `tn` 或 `both`，会因为公开库缺少 TN 标签而报错。
English: result as `tn` or `both`, will because start database missing TN label raise an error.

### 5.2 用 PowerShell 检查数据库是否存在 / 5.2 use PowerShell checkdata database is no in.

在 `Training Code` 目录执行：
English: in `Training Code` execute:

```powershell
Test-Path "..\PublicSoilSampleDatabase\public_dataset_manifest.json"
Test-Path "..\PublicSoilSampleDatabase\samples"
(Get-ChildItem "..\PublicSoilSampleDatabase\samples" -Filter *.npz).Count
```

当前公开库期望样本数为：
English: current start database sample count as:

```text
4372
```

如果 `Test-Path` 返回 `False`，说明数据库不在默认位置。可以用命令临时指定数据库根目录：
English: result `Test-Path` return `False`, notedata database not in default. can use temporary data database root directory:

```powershell
python Train_main.py --dataset-root "D:\YourPath\PublicSoilSampleDatabase" --dry-run
```

### 5.3 用 Python 读取一个样本 / 5.3 use Python read sample.

这个命令只读取一个 `.npz`，用于确认 Python 能打开数据库：
English: onlyread `.npz`, use confirm confirm Python can start data database:

```powershell
python -c "from pathlib import Path; import numpy as np; root=Path('..')/'PublicSoilSampleDatabase'; p=next((root/'samples').glob('*.npz')); data=np.load(p, allow_pickle=False); print(p.name); print(data['hyper'].shape, data['nir'].shape, data['image'].shape, data['target_names'].tolist(), data['labels']); data.close()"
```

期望结构大致为：
English: result large as:

```text
XXXXXXXX.npz
(681,) (5,) (8, 1024, 1024) ['SOC'] [SOC数值]
```

## 6. 第一次运行：只检查配置，不训练 / section run: only checks the configuration, does not train.

### 6.1 查看可选菜单 / 6.1 optional single.

```powershell
python Train_main.py --list-menus
```

当前主要菜单：
English: current need single:

| 菜单 key<br>EN: single key. | 作用<br>EN: Purpose: | 新手建议<br>EN: beginnerrecommended. |
| --- | --- | --- |
| `mfpchfnetv2` | MFPC-HFNet 主模型和结构消融。<br>EN: MFPC-HFNet and architecture ablation. | 默认先用这个。<br>EN: default first use. |
| `input_ablation` | 输入端消融，例如 NIR-only、Hyper-only、Image+NIR。<br>EN: input-side ablation, for example NIR-only, Hyper-only, Image+NIR. | 有 Full 参考结果后再用。<br>EN: Full result result after then use. |
| `compare` | ResNet、EfficientNet、ViT、Swin、ConvNeXt、MobileNetV4 等 backbone 对比。<br>EN: ResNet, EfficientNet, ViT, Swin, ConvNeXt, MobileNetV4 backbone for. | 计算量大，熟悉后再用。<br>EN: amount large, after then use. |

### 6.2 查看某个菜单有哪些模型 / 6.2 single.

```powershell
python Train_main.py --menu mfpchfnetv2 --list-models
```

默认主模型是：
English: default is:

```text
MFPCHFNetV2_Full
```

### 6.3 打印默认训练配置 / 6.3 print defaulttrain.

```powershell
python Train_main.py --dry-run
```

重点检查输出摘要中的这些字段：
English: check need in field:

| 字段<br>EN: field. | 应该重点看什么<br>EN: should this. |
| --- | --- |
| `菜单接口`<br>EN: ``. | 是否为 `mfpchfnetv2`。<br>EN: is no as `mfpchfnetv2`. |
| `训练模型`<br>EN: ``. | 是否只包含你想训练的模型。<br>EN: is no only want train. |
| `TARGET_MODE` | 公开版应为 `soc`。<br>EN: public release should as `soc`. |
| `DATASET_ROOT` | 是否指向公开数据库根目录。<br>EN: is no start data database root directory. |
| `DATA_DIR` | 是否指向 `PublicSoilSampleDatabase\samples`。<br>EN: is no `PublicSoilSampleDatabase\samples`. |
| `PCA_PRIORS_PATH` | 是否指向 `ModelAssets\pca_priors_full.pt`。<br>EN: is no `ModelAssets\pca_priors_full.pt`. |
| `NUM_FOLDS` / `NUM_RUNS` | 正式训练默认 8 / 8；调试时可临时改小。<br>EN: formal trainingdefault 8 / 8; when can temporary change small. |
| `MODEL_DATA_DIR` | 输出会写到哪里。<br>EN: will write to. |
| `EXPORT_ONNX_AFTER_TRAINING` | 是否训练后自动导出 ONNX。<br>EN: is no train after automaticallyexport ONNX. |

如果 dry-run 已经报错，先不要正式训练。优先解决路径、依赖或参数问题。
English: result dry-run already raise an error, first do notformal training.prefer path, or parameters.

## 7. 冒烟测试：确认能训练一个很小任务 / smoke test: confirm confirm can train small task task.

冒烟测试的目标是确认环境、数据库、模型构建、Fold 构建、PCA 先验、训练循环和输出目录都能跑通。它不是正式结果，不能用于论文或部署。
English: smoke test is confirm confirm environment environment, data database, build, Fold build, PCA first, train environment and output directory all can pass. not is formal result result, not can use or.

推荐命令：
English: :

```powershell
python Train_main.py --menu mfpchfnetv2 --train-model-names MFPCHFNetV2_Full --num-runs 1 --max-epochs 1 --export-onnx-after-training false
```

这条命令含义：
English: meaning:

| 参数<br>EN: parameters. | 含义<br>EN: Meaning: |
| --- | --- |
| `--menu mfpchfnetv2` | 使用 MFPC-HFNet 菜单。<br>EN: use MFPC-HFNet single. |
| `--train-model-names MFPCHFNetV2_Full` | 只训练 Full 主模型。<br>EN: onlytrain Full. |
| `--num-runs 1` | 只跑 Fold01 对应的 1 个 run。<br>EN: only Fold01 for should 1 run. |
| `--max-epochs 1` | 只训练 1 个 epoch。<br>EN: onlytrain 1 epoch. |
| `--export-onnx-after-training false` | 冒烟测试不导出 ONNX，减少额外依赖和时间。<br>EN: smoke test not export ONNX, fewer and when. |

如果冒烟测试成功，你应该能在 `ModelData/` 下看到类似目录：
English: result smoke test, should this can in `ModelData/` below to:

```text
ModelData/
  2026-xx-xx_xx-xx-xx_mfpchfnetv2_SOC_8FoldCV/
    MFPCHFNetV2_Full/
      Fold01/
        metrics_summary.csv
        test_predictions_fold_01_soc.csv
        validation_history.csv
        training_progress.json
        run_info.json
        best_model.pth
```

注意：`NUM_FOLDS` 仍然可以保持 8，`NUM_RUNS=1` 只表示本次只运行其中 1 个 Fold。
English: : `NUM_FOLDS` still can 8, `NUM_RUNS=1` onlymeans onlyrun its in 1 Fold.

## 8. 正式训练怎么启动 / formal training start.

### 8.1 默认正式训练 / 8.1 defaultformal training.

确认 dry-run 没问题后，正式训练命令是：
English: confirm confirm dry-run not after, formal training is:

```powershell
python Train_main.py
```

当前 `Train_main.py` 默认设置：
English: current `Train_main.py` default:

```text
TRAIN_MENU = "mfpchfnetv2"
TRAIN_MODEL_NAMES = ["MFPCHFNetV2_Full"]
TARGET_MODE = "soc"
DATASET_ROOT = Training Code 上一级/PublicSoilSampleDatabase
DATA_DIR = DATASET_ROOT/samples
GT_PATH = ""
TN_PATH = ""
PCA_PRIORS_PATH = ModelAssets/pca_priors_full.pt
RESUME_TRAINING = False
MODEL_DATA_DIR = ModelData
NUM_FOLDS = 8
NUM_RUNS = 8
VALIDATION_FOLD_OFFSET = 1
SPLIT_SEED = 20260317
EXPORT_ONNX_AFTER_TRAINING = True
```

正式训练启动后，终端会先打印最终配置摘要，然后进入训练。训练过程中会按模型和 Fold 逐个输出日志。
English: formal trainingstart after, terminal will first print most need, after train.train program in will by and Fold.

### 8.2 常用命令模板 / 8.2 use.

只训练默认 Full，但不导出 ONNX：
English: onlytraindefault Full, not export ONNX:

```powershell
python Train_main.py --export-onnx-after-training false
```

只跑前 1 个 Fold，用于较长调试：
English: only before 1 Fold, use:

```powershell
python Train_main.py --num-runs 1
```

临时减少 epoch：
English: temporary fewer epoch:

```powershell
python Train_main.py --max-epochs 50
```

换输出目录：
English: output directory:

```powershell
python Train_main.py --model-data-dir "D:\SoilTrainingRuns\ModelData"
```

换公开数据库目录：
English: start data database:

```powershell
python Train_main.py --dataset-root "D:\Datasets\PublicSoilSampleDatabase"
```

同时换数据库和输出目录，并先 dry-run：
English: same when data database and output directory, and first dry-run:

```powershell
python Train_main.py --dataset-root "D:\Datasets\PublicSoilSampleDatabase" --model-data-dir "D:\SoilTrainingRuns\ModelData" --dry-run
```

训练 MFPC-HFNet 结构消融：
English: train MFPC-HFNet architecture ablation:

```powershell
python Train_main.py --menu mfpchfnetv2 --train-model-preset ablation_only
```

训练某几个指定结构：
English: train result:

```powershell
python Train_main.py --menu mfpchfnetv2 --train-model-names MFPCHFNetV2_H3Low,MFPCHFNetV2_LowOnly
```

### 8.3 不建议新手直接修改源码的参数 / 8.3 not recommendedbeginnerdirectly change source code parameters.

`Train_main.py` 是用户手工参数面板。修改该文件中的学习率、epoch、batch size、weight decay、断点目录、数据路径或模型选择前，应先列出：
English: `Train_main.py` is use manualparameters. change this file in learning rate, epoch, batch size, weight decay,, datapath or select before, should first column:

```text
拟修改参数
当前值
拟修改值
修改原因
预期影响
```

新手优先使用命令行临时覆盖，例如：
English: beginnerpreferuse line temporary override, for example:

```powershell
python Train_main.py --learning-rate 0.0001 --max-epochs 100 --dry-run
```

命令行覆盖的优点是不会改坏源码，也便于对比不同实验。
English: line override is not will change source code, also to make it easier to for not same.

## 9. 训练输出怎么看 / train.

默认输出根目录：
English: default root directory:

```text
Training Code/ModelData/
```

一次训练会生成一个时间戳实验目录：
English: train will generate when:

```text
ModelData/
  2026-xx-xx_xx-xx-xx_mfpchfnetv2_SOC_8FoldCV/
```

典型输出结构：
English: result:

```text
ModelData/
  2026-xx-xx_xx-xx-xx_mfpchfnetv2_SOC_8FoldCV/
    mfpchfnetv2_unified_train_plan.json
    MFPCHFNetV2_Full/
      auto_batch_size_plan.json
      Fold01/
        best_model.pth
        metrics_summary.csv
        test_predictions_fold_01_soc.csv
        validation_history.csv
        training_progress.json
        run_info.json
        pca_priors_train_only.pt
        pca_priors_train_only_summary.json
      Fold02/
        ...
      final_test_metrics_value_pm_std.csv
      test_predictions_all_folds.csv
      ONNX/
        DLV-3-Both-Full.onnx
        DLV-3-Both-Full.onnx.export_info.json
```

关键文件说明：
English: filenote:

| 文件<br>EN: file. | 位置<br>EN: English counterpart for the Chinese technical note above. | 含义<br>EN: Meaning: |
| --- | --- | --- |
| `mfpchfnetv2_unified_train_plan.json` | 实验目录根部<br>EN: English counterpart for the Chinese technical note above. | 本次训练计划，记录模型清单和关键配置。<br>EN: train, model list and. |
| `auto_batch_size_plan.json` | 单模型目录<br>EN: single. | 记录菜单 batch、实际 batch、CUDA 显存预检和自动降 batch 结果。<br>EN: single batch, batch, CUDA and automatically batch result result. |
| `best_model.pth` | Fold 目录<br>EN: Fold. | 当前 Fold 验证集最优权重。<br>EN: current Fold validation set most. |
| `resume_from_best_state.pth` | Fold 目录<br>EN: Fold. | 断点续训状态；Fold 完成后可能按清理策略删除。<br>EN: resume training; Fold complete after can can by clean up. |
| `training_progress.json` | Fold 目录<br>EN: Fold. | 当前 Fold 训练进度、最佳 epoch、学习率和 batch。<br>EN: current Fold train degree, most epoch, learning rate and batch. |
| `validation_history.csv` | Fold 目录<br>EN: Fold. | 每次验证的训练和验证指标历史。<br>EN: each validate train and validatemetrics. |
| `metrics_summary.csv` | Fold 目录<br>EN: Fold. | 当前 Fold 的 Train/Validation/Test 指标摘要，也是 Fold 完成标志。<br>EN: current Fold Train/Validation/Test metrics need, also is Fold complete. |
| `test_predictions_fold_XX_soc.csv` | Fold 目录<br>EN: Fold. | 当前 Fold 测试集逐样本预测值。<br>EN: current Fold test set sample value. |
| `run_info.json` | Fold 目录<br>EN: Fold. | 当前 Fold 的设备、数据、模型、PCA 先验和参数追溯。<br>EN: current Fold, data,, PCA first and parameters. |
| `pca_priors_train_only.pt` | Fold 目录<br>EN: Fold. | 只用当前 Fold 的 Train 子集构建的 PCA 先验。<br>EN: only use current Fold Train build PCA first. |
| `final_test_metrics_value_pm_std.csv` | 单模型目录<br>EN: single. | 所有已完成 Fold 的 Test 指标均值和标准差。<br>EN: already complete Fold Test metricsmean and standard deviation. |
| `test_predictions_all_folds.csv` | 单模型目录<br>EN: single. | 所有已完成 Fold 的测试集预测汇总。<br>EN: already complete Fold test set. |
| `ONNX/*.onnx` | 单模型目录<br>EN: single. | 训练后导出的应用端模型。<br>EN: train after export should use. |
| `ONNX/*.export_info.json` | 单模型目录<br>EN: single. | ONNX 来源 Fold、输入输出接口、参数量、数据库追溯等信息。<br>EN: ONNX source Fold, inputs and outputs interface, count, data database. |

### 9.1 哪个文件最适合看最终效果 / 9.1 file most suitable for most result.

论文表格或整体性能优先看：
English: table or property can prefer:

```text
final_test_metrics_value_pm_std.csv
```

逐样本预测误差优先看：
English: sample prefer:

```text
test_predictions_all_folds.csv
```

单个 Fold 是否成功完成优先看：
English: single Fold is no completeprefer:

```text
FoldXX/metrics_summary.csv
```

模型部署优先看：
English: prefer:

```text
ONNX/DLV-3-Both-Full.onnx
ONNX/DLV-3-Both-Full.onnx.export_info.json
```

### 9.2 Fold 完成和续训判断 / 9.2 Fold complete and determine.

当前工程把下面文件作为 Fold 完成标志：
English: current program below file as Fold complete:

```text
FoldXX/metrics_summary.csv
```

如果某个 Fold 已经有 `metrics_summary.csv`，续训时训练引擎会认为这个 Fold 已完成，不会重复训练该 Fold。
English: result Fold already `metrics_summary.csv`, when training engine will confirm as Fold already complete, not will train this Fold.

## 10. 断点续训 / resume training.

### 10.1 什么时候需要续训 / 10.1 when need need.

这些情况通常需要断点续训：
English: usually need need resume training:

- 电脑重启。
English: computer computer.
- 训练被手动停止。
English: train manual.
- 显存或磁盘问题修复后继续。
English: or after.
- 只完成了一部分 Fold，想接着跑剩余 Fold。
English: onlycomplete Fold, want Fold.

### 10.2 用命令行续训 / 10.2 use line.

假设旧实验目录是：
English: old is:

```text
C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code\ModelData\2026-xx-xx_xx-xx-xx_mfpchfnetv2_SOC_8FoldCV
```

续训命令：
English: :

```powershell
python Train_main.py --resume-training true --resume-save-dir "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code\ModelData\2026-xx-xx_xx-xx-xx_mfpchfnetv2_SOC_8FoldCV"
```

续训注意事项：
English: notes:

- `--resume-save-dir` 必须指向实验目录根部，不是某个 Fold 目录。
English: `--resume-save-dir` must, not is Fold.
- 已经存在 `metrics_summary.csv` 的 Fold 会跳过。
English: already in `metrics_summary.csv` Fold will skip.
- 未完成 Fold 会优先读取 `resume_from_best_state.pth` 和 `training_progress.json`。
English: not yet complete Fold will preferread `resume_from_best_state.pth` and `training_progress.json`.
- 续训时不要随意改 `SPLIT_SEED`、`NUM_FOLDS`、模型清单或数据库，否则旧结果和新结果可能不再可比。
English: when do not change `SPLIT_SEED`, `NUM_FOLDS`, model list or data database, no then old result result and new result result can can no longer can.

## 11. 菜单与模型清单 / single and model list.

### 11.1 MFPC-HFNet 结构菜单 / 11.1 MFPC-HFNet result single.

文件：`Menu_MFPCHFNetV2.py`
English: file: `Menu_MFPCHFNetV2.py`.

| 菜单模型名<br>EN: single name. | 展示名<br>EN: name. | 结构<br>EN: result. | 图像尺寸<br>EN: image. |
| --- | --- | --- | --- |
| `MFPCHFNetV2_Full` | `MFPC-HFNet` | high1+high2+high3+low | 1024x1024 |
| `MFPCHFNetV2_H2H3Low` | `MFPC-HFNet-H2H3Low` | high2+high3+low | 512x512 |
| `MFPCHFNetV2_H3Low` | `MFPC-HFNet-H3Low` | high3+low | 256x256 |
| `MFPCHFNetV2_LowOnly` | `MFPC-HFNet-LowOnly` | low | 128x128 |

该菜单负责 MFPC-HFNet 结构消融。`optimizer_policy="layerwise_lr"` 表示融合/跨频层交互相关参数组使用较低学习率，具体参数分组由 `Model_MFPCHFNet.py` 提供，执行由 `Train_optimizer.py` 负责。
English: this single MFPC-HFNet architecture ablation.`optimizer_policy="layerwise_lr"` meansfusion/ frequency band parameters use low learning rate, parameters by `Model_MFPCHFNet.py`, execute by `Train_optimizer.py`.

### 11.2 输入端消融菜单 / 11.2 input-side ablation single.

文件：`Menu_InputAblation.py`
English: file: `Menu_InputAblation.py`.

训练组合：
English: train:

- `InputAblation_NIROnly`
- `InputAblation_HyperOnly`
- `InputAblation_HyperNIR`
- `InputAblation_ImageOnly`
- `InputAblation_ImageNIR`
- `InputAblation_ImageHyper`

Full 基线不进入该菜单训练队列，仅作为结果对照的只读参考来源。没有图像输入的组合不构建图像分支，DataLoader 也不读取无关图像张量。
English: Full not this single train column, only as result result for only read source. not image not buildimage, DataLoader also not read no image tensor.

输入端消融如果要严格复用 Full 的折分，需要提供 Full 参考实验中的：
English: input-side ablation result need reuse Full, need need Full in:

```text
shared_folds/fold_assignments.csv
```

如果没有该文件，可临时关闭外部 shared folds 读取，让代码按 `SPLIT_SEED` 重新构建折分：
English: result not this file, can temporary shared folds read, let code by `SPLIT_SEED` new build:

```powershell
python Train_main.py --menu input_ablation --load-shared-folds-from-csv false --dry-run
```

### 11.3 Backbone 对比菜单 / 11.3 Backbone for single.

文件：`Menu_Compare_AllBackbones.py`
English: file: `Menu_Compare_AllBackbones.py`.

该菜单包含 ResNet、ResNeXt、EfficientNet、EfficientNetV2、ViT、Swin Transformer V2、ConvNeXt、MobileNetV4 等 baseline/backbone。各条目只声明 backbone 名称、分辨率、显示名和训练微调标签；训练循环仍由 `Train_core.py` 统一执行。
English: this single ResNet, ResNeXt, EfficientNet, EfficientNetV2, ViT, Swin Transformer V2, ConvNeXt, MobileNetV4 baseline/backbone. each only backbone name,, name and train label; train environment still by `Train_core.py` execute.

Backbone 对比计算量大，新手建议先跑 `mfpchfnetv2` 的 dry-run 和冒烟测试，再尝试 compare 菜单。
English: Backbone for amount large, beginnerrecommended first `mfpchfnetv2` dry-run and smoke test, then compare single.

## 12. 交叉验证与数据使用 / cross-validation and datause.

主训练流程的 Fold 构建入口为 `Train_core.py` 中的 `build_fold_assignments_for_training()`，默认调用 `Train_support.py` 的稳定 ID 折分逻辑。Fold 构建仅使用数据索引中的稳定样本 ID。
English: train program Fold build interface as `Train_core.py` in `build_fold_assignments_for_training()`, default use `Train_support.py` stable ID logic.Fold buildonlyusedataindexing in stablesample ID.

训练数据使用流程：
English: traindatause program:

```text
Train_main.py
  -> Train_config.py 装配菜单
  -> Train_core.py 构建 Dataset
  -> Data_LoaderRuntimeAuto.py 识别公开库并建立样本索引
  -> Train_support.py 生成 Fold 索引
  -> Train_core.py 为 train/val/test 构建 DataLoader
  -> Dataset.__getitem__ 按索引懒加载 .npz 中的 image/hyper/nir/label
```

每个 run 的集合划分语义：
English: each run:

- `Test`：当前 run 对应的 Fold。
English: `Test`: current run for should Fold.
- `Validation`：相对 Test 偏移 `VALIDATION_FOLD_OFFSET` 的 Fold。
English: `Validation`: for Test `VALIDATION_FOLD_OFFSET` Fold.
- `Train`：其余 Fold。
English: `Train`: its Fold.

对于模型未启用的输入源，Data 层和 DataLoader 会尽量裁剪无关张量，避免 NIR-only、Hyper-only 等实验在 CPU 和 GPU 端搬运无关图像。
English: for not yet enable input sources, Data and DataLoader will amount no amount, avoid NIR-only, Hyper-only in CPU and GPU no image.

## 13. MFPC-HFNet 模型设计 / MFPC-HFNet.

论文正式名称为 **MFPC-HFNet**，英文全称为 **Multi-Frequency Principal Component Hierarchical Fusion Network**，中文名称为 **主成分层级融合网络**。代码中使用 `MFPCHFNet` 作为工程标识。
English: formalname as **MFPC-HFNet**, full as **Multi-Frequency Principal Component Hierarchical Fusion Network**, in name as **principal components fusion **. code in use `MFPCHFNet` as program.

模型面向 8 通道土壤图像、681 维 HyperVISNIR 和 5 维 NIR 数值输入。Full 主模型输入为：
English: 8 pass soil soil image, 681 HyperVISNIR and 5 NIR value.Full as:

- `image`: `[B, 8, 1024, 1024]`
- `hyper`: `[B, 681]`
- `nir`: `[B, 5]`

输出维度由训练目标决定。当前公开版为 SOC 单目标，输出维度为 1。
English: degree by train.currentpublic release as SOC single, degree as 1.

### 13.1 两阶段结构 / 13.1 result.

MFPC-HFNet 包含两个阶段：
English: MFPC-HFNet:

1. 先验构建阶段：训练时由 `Data_BuildPcaPriorsFull.py` 按当前 Fold 的 Train 子集构建 `pca_priors_train_only.pt`。
English: first build: train when by `Data_BuildPcaPriorsFull.py` by current Fold Train build `pca_priors_train_only.pt`.
2. 训练学习阶段：模型读取当前 Fold 的 PCA 先验，将归一化和 PCA 投影嵌入图像分支前端，再通过 PCASE、LD Encoder、HFSA、HLAF 和多模态融合头完成回归。
English: train learning learning: readcurrent Fold PCA first, and PCA image before, then pass PCASE, LD Encoder, HFSA, HLAF and more modalityfusion complete.

公开版保留一份默认全局先验文件，主要用于兼容旧结果或非正式检查：
English: public releasekeep default full first file, need use compatible old result result or non-formalcheck:

```text
ModelAssets/pca_priors_full.pt
```

正式交叉验证训练中，每个 Fold 会在对应输出目录下写出：
English: formalcross-validationtrain in, each Fold will in for should output directory below write:

```text
FoldXX/pca_priors_train_only.pt
FoldXX/pca_priors_train_only_summary.json
```

该 Fold 先验只由当前 Train 子集估计，Validation/Test 样本不参与通道归一化、结构向量筛选或 PCA 参数估计。
English: this Fold first only by current Train estimate, Validation/Test sample not participate inchannel normalization, structural-vector selection or PCA parametersestimate.

为减少输入消融等多模型训练中的重复计算，同一实验输出根目录还会维护 Fold 级共享先验缓存：
English: as fewer more train in, same root directory will maintain Fold first cache:

```text
_shared_pca_priors/<shared_prior_cache_key>/pca_priors_train_only.pt
_shared_pca_priors/<shared_prior_cache_key>/pca_priors_train_only_summary.json
```

共享缓存命中后，训练引擎会把共享先验复制回当前模型的 `FoldXX/` 目录，再按当前模型写入自己的 summary。共享 key 不包含模型名和 active_inputs，因此 `ImageOnly`、`Image+NIR`、`Image+Hyper` 这类图像先验相同的模型可复用；但 key 包含样本身份、Train 子集、图像尺寸、频层结构和 PCA 构建参数，不会跨不同结构、不同分辨率或不同划分误用。
English: shared cache in after, training engine will first current `FoldXX/`, then by current write summary. key not name and active_inputs, therefore `ImageOnly`, `Image+NIR`, `Image+Hyper` image first same can reuse; key sample, Train, image, frequency band result and PCA buildparameters, not will not same result, not same or not same use.

### 13.2 频层拓扑 / 13.2 frequency band.

Full 模型对 1024x1024 图像构建 3 层拉普拉斯金字塔：
English: Full for 1024x1024 imagebuild 3 Laplacian pyramid:

| 频层<br>EN: frequency band. | 原始频层尺寸<br>EN: start frequency band. | PCASE 后 source map<br>EN: PCASE after source map. | 8x8 patch 后 token grid<br>EN: 8x8 patch after token grid. |
| --- | --- | --- | --- |
| high1 | 1024x1024 | 64x64 | 8x8 |
| high2 | 512x512 | 32x32 | 4x4 |
| high3 | 256x256 | 16x16 | 2x2 |
| low | 128x128 | 8x8 | 1x1 |

H2H3Low、H3Low 和 LowOnly 不是简单缩放 Full，而是按最高保留频层构建对应金字塔路径。
English: H2H3Low, H3Low and LowOnly not is single Full, is by most high keepfrequency bandbuild for should path.

### 13.3 PCASE

PCASE 表示 **Principal Component Adaptive Scale Embedding module**，中文为 **主成分自适应尺度嵌入模块**。它位于固定 PCA 投影之后、局部动态编码之前，负责根据主成分贡献度和有效结构向量占比分配通道容量。
English: PCASE means **Principal Component Adaptive Scale Embedding module**, in as **principal components should degree **. fixed PCA after, programming code before, data principal components degree and structural vectors pass amount.

普通文本公式如下，便于复制到 Word 后直接编辑：
English: pass form below, to make it easier to to Word after directly programming:

```text
effective_pca_rank = (sum(eigvals)^2) / sum(eigvals^2)

S_l = input_size_l / target_size_l

high_capacity_raw_l = effective_pca_rank_l * S_l^2 * N_l
low_capacity_raw = effective_pca_rank_low * S_low * N_low

total_out_channels_l = max(K_l * min_dim, ceil(capacity_raw_l))
```

其中 `K_l` 为第 l 个频层保留的 PCA 主成分数，`S_l` 为空间边长压缩倍率，`N_l` 为有效结构向量占比。high1、high2、high3 使用面积补偿规则；low 使用类卷积式通道增长，避免低频层通道规模异常膨胀。
English: its in `K_l` as section l frequency bandkeep PCA principal components number, `S_l` as compression ratio, `N_l` as structural vectors.high1, high2, high3 use rule; low use form pass, avoidlow-frequency level pass.

### 13.4 Token 宽度 / 13.4 Token degree.

PCASE 后的局部块容量用于推导 token 维度：
English: PCASE after local block amount use token degree:

```text
source_flat_dim_l = patch_size^2 * pcase_channels_l
raw_token_dim_l = source_flat_dim_l / token_compression_ratio
token_dim_l = max(token_dim_min, ceil_to_multiple(raw_token_dim_l, token_dim_round_multiple))
```

当前默认：
English: currentdefault:

- `patch_size = 8`
- `token_compression_ratio = 8`
- `token_dim_min = 96`
- `token_dim_round_multiple = 16`

### 13.5 层级融合 / 13.5 fusion.

高频摘要由 HFSA 路径完成：
English: high-frequency summary by HFSA pathcomplete:

```text
H1 token grid -> PixelUnshuffle(2) -> H2 融合 -> CrossPatchEncoder -> HF2
HF2 -> PixelUnshuffle(2) -> H3 融合 -> CrossPatchEncoder -> HF summary
```

高低频融合由 HLAF 路径完成：
English: high low fusion by HLAF pathcomplete:

```text
HF summary [B, Dhf, 2, 2] -> PixelUnshuffle(2) -> 高频父级摘要 [B, 4*Dhf, 1, 1]
高频父级摘要 + low token [B, Dlow, 1, 1] -> CrossPatchEncoder -> image token
```

结构图中可将 H1->H2、HF2->H3 和 HF summary->low 三段父子尺度对齐融合概括为 **AFU**，即 **Aligned Fusion Unit**。其中前两段 AFU 属于 HFSA，最后一段属于 HLAF。
English: result image in can H1->H2, HF2->H3 and HF summary->low degree alignmentfusion as **AFU**, **Aligned Fusion Unit**. its in before AFU property HFSA, most after property HLAF.

## 14. 数据库重构 / data database.

大多数公开版使用者不需要重构数据库。只有在你要从旧原始样本库或已校正缓存重新生成公开库时，才需要使用本节。
English: large more number public releaseuse not need need data database.only in need from old start sample database or already cache new generate start database when, need need use section.

公开库重构脚本为：
English: start database as:

```text
Data_RebuildPublicSampleDatabase.py
```

它可以从已校正缓存重新打包，也可以从旧原始样本库扫描后执行 blank 校正再写出公开库。
English: can from already cache new, also can from old start sample database scan after execute blank then write start database.

推荐命令模板：
English: :

```powershell
python Data_RebuildPublicSampleDatabase.py `
  --label-xlsx ".\ExampleData\labels.xlsx" `
  --cache-dir ".\ExampleData\calibrated_cache" `
  --output-root "$env:USERPROFILE\Desktop\PublicSoilSampleDatabase"
```

常用参数：
English: use parameters:

| 参数<br>EN: parameters. | 含义<br>EN: Meaning: |
| --- | --- |
| `--label-xlsx` | SOC 标签表，至少包含 `SampleName` 和 SOC 数值列。<br>EN: SOC label table, contains at least `SampleName` and SOC value column. |
| `--cache-dir` | 已校正缓存目录，优先用于快速重打包。<br>EN: already cache, prefer using. |
| `--source-root` / `--source-data-dir` | 无缓存时用于扫描旧原始库。<br>EN: no cache when use scan old start database. |
| `--output-root` | 公开库输出目录。<br>EN: start database output directory. |
| `--compressed` | 使用压缩 `.npz`，体积更小但写入更慢。<br>EN: use `.npz`, small write. |
| `--limit` | 只写出前 N 个样本，适合接口冒烟测试。<br>EN: only write before N sample, suitable for interface smoke test. |
| `--overwrite` | 允许覆盖已有输出目录。<br>EN: allowoverride already output directory. |

重构脚本只在一次性数据发布阶段使用旧样本名归一化逻辑。公开版训练阶段不再执行旧样本名转换，也不会输出旧样本名。
English: only in property data use old sample name logic.public releasetrain no longerexecute old sample name, also not will old sample name.

## 15. 验证命令 / validate.

发布或交给别人使用前，建议至少运行下面三类检查。
English: or use before, recommended fewer run below check.

### 15.1 Python 语法检查 / 15.1 Python check.

```powershell
python -B -m py_compile Train_main.py Train_config.py Train_core.py Train_export_onnx.py Train_optimizer.py Train_support.py Metrics_core.py Menu_MFPCHFNetV2.py Menu_InputAblation.py Menu_Compare_AllBackbones.py Tool\Tool_CheckTrainingPolicy.py Model_MFPCHFNet.py Model_CompareBackbones.py Model_EfficientNet1024Backbones.py Data_LoaderRuntimeAuto.py Data_DiskCacheRegistry.py Data_BuildPcaPriorsFull.py Data_PublicSampleDatabase.py Data_RebuildPublicSampleDatabase.py
```

### 15.2 工程边界静态检查 / 15.2 engineering boundary check.

```powershell
python Tool\Tool_CheckTrainingPolicy.py
```

期望输出：
English: :

```text
V2 training policy check PASSED.
```

### 15.3 训练配置检查 / 15.3 train check.

```powershell
python Train_main.py --dry-run
```

公开库接口冒烟测试可使用 `Data_RebuildPublicSampleDatabase.py --limit 2` 先生成一个小型临时库，再用 `Train_main.py --dry-run` 或自定义 Dataset 读取脚本确认数据字段。
English: start database interface smoke test can use `Data_RebuildPublicSampleDatabase.py --limit 2` first generate small temporary database, then use `Train_main.py --dry-run` or Dataset read confirm confirm datafield.

## 16. 常见问题排查 / English counterpart for the Chinese technical note above.

### 16.1 `ModuleNotFoundError: No module named 'cv2'`

原因：缺少 OpenCV 的 Python 包。
English: : missing OpenCV Python.

解决：
English: :

```powershell
python -m pip install opencv-python
```

注意：导入名是 `cv2`，安装包名是 `opencv-python`。
English: : name is `cv2`, name is `opencv-python`.

### 16.2 `torch.cuda.is_available()=False`

原因通常是：
English: usually is:

- 安装了 CPU 版 PyTorch。
English: CPU PyTorch.
- 当前虚拟环境不是安装 CUDA 版 PyTorch 的环境。
English: current environment environment not is CUDA PyTorch environment environment.
- NVIDIA 驱动不可用。
English: NVIDIA not can use.
- CUDA / PyTorch 版本不匹配。
English: CUDA / PyTorch not.

先检查：
English: first check:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

正式训练要求 CUDA 可用。不要为了绕过报错直接改成 CPU 训练，除非只是很小的开发冒烟测试。
English: formal training need CUDA can use.do not as raise an errordirectly change CPU train, non-only is small start smoke test.

### 16.3 找不到公开数据库 / 16.3 find not to start data database.

先检查默认位置：
English: first checkdefault:

```powershell
Test-Path "..\PublicSoilSampleDatabase\public_dataset_manifest.json"
Test-Path "..\PublicSoilSampleDatabase\samples"
```

如果数据库在其他位置，用：
English: result data database in its, use:

```powershell
python Train_main.py --dataset-root "D:\Datasets\PublicSoilSampleDatabase" --dry-run
```

### 16.4 `TARGET_MODE` 设置为 `tn` 或 `both` 后报错 / 16.4 `TARGET_MODE` as `tn` or `both` after raise an error.

当前公开数据库只发布 SOC 标签，没有 TN 标签。公开版应保持：
English: current start data database only SOC label, not TN label.public release should:

```text
TARGET_MODE = "soc"
```

### 16.5 训练中 CUDA out of memory / 16.5 train in CUDA out of memory.

当前训练引擎有 CUDA 显存预检，会在训练前尝试降低 batch。如果仍然 OOM，优先考虑：
English: currenttraining engine CUDA, will in train before low batch. result still OOM, prefer:

- 关闭其他占用 GPU 的程序。
English: its use GPU program order.
- 临时只跑较小模型或较低分辨率结构。
English: temporaryonly small or low result.
- 临时减少 `NUM_RUNS` 做分批训练。
English: temporary fewer `NUM_RUNS` do train.
- 把输出目录和缓存目录放到空间更充足的磁盘。
English: output directory and cache to meet.

不要直接在 `Train_core.py` 中硬改显存策略。优先通过菜单和命令行控制实验规模。
English: do notdirectly in `Train_core.py` in change.prefer pass single and line control.

### 16.6 训练中断后不知道是否完成 / 16.6 train in after not is no complete.

查看每个 Fold 是否存在：
English: each Fold is no in:

```text
FoldXX/metrics_summary.csv
```

存在这个文件表示该 Fold 已完成。续训时会跳过已完成 Fold。
English: in filemeans this Fold already complete. when will skip already complete Fold.

### 16.7 PowerShell 路径有空格或中文导致命令失败 / 16.7 PowerShell path or in.

路径必须加英文双引号：
English: pathmust English double quotes:

```powershell
Set-Location "C:\Users\94562\Desktop\Multisource Data LJY 2025\Training Code"
```

如果终端中文显示异常，可先设置 UTF-8 输出：
English: result terminal in, can first UTF-8:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
```

### 16.8 ONNX 导出失败但训练已完成 / 16.8 ONNX export train already complete.

先看模型目录下是否已有：
English: first below is no already:

```text
final_test_metrics_value_pm_std.csv
test_predictions_all_folds.csv
FoldXX/metrics_summary.csv
```

如果这些存在，说明训练主体结果已经写出。ONNX 导出失败通常与 `onnx`、`thop`、模型权重选择或导出接口有关，可以先用：
English: result in, notetrain result result already write.ONNX export usually and `onnx`, `thop`, select or export interface, can first use:

```powershell
python Train_main.py --export-onnx-after-training false
```

完成训练主体，再单独排查导出问题。
English: completetrain, then single export.

## 17. 发布边界 / English counterpart for the Chinese technical note above.

公开发布时应包含：
English: start when should:

- 源码 `.py`
English: source code `.py`.
- `README.md`
- `.gitignore`
- `ModelAssets/pca_priors_full.pt`
- 已重构的公开数据库，或公开数据库下载说明
English: already start data database, or start data database below note.

公开发布时不应包含：
English: start when not should:

- `ModelData/`
- 原始数据库
English: start data database.
- 原始标签 Excel
English: start label Excel.
- `.mat` 真值表
English: `.mat` value table.
- `__pycache__/`
- 临时 smoke 测试库
English: temporary smoke test database.
- 训练 checkpoint、ONNX、Tensor/engine 等大体积训练产物，除非本次发布目标明确要求包含
English: train checkpoint, ONNX, Tensor/engine large train, non- confirm need.

`.gitignore` 已按上述边界排除常见本地输出和私有数据，但手动打包发布时仍应以本节为准检查。
English: `.gitignore` already by on and data, manual when still should section as check.

## 18. 维护规则 / maintainrule.

- 菜单只声明训练意图和模型规格，不读取数据、不保存 checkpoint、不导出指标。
English: single only train image and model specification, not readdata, not save checkpoint, not exportmetrics.
- 训练循环只能位于 `Train_core.py`，不得新增多个同级训练主体。
English: train environment only can `Train_core.py`, must not new more same train.
- 模型结构只能位于 `Model_*.py`，菜单通过 `ModelSpec` 和菜单 `Config` 显式传参。
English: model architectureonly can `Model_*.py`, single pass `ModelSpec` and single `Config` form.
- 数据读取、公开库重构、blank 校正和缓存管理应放在 `Data_*.py`。
English: data loading, start database, blank and cache should in `Data_*.py`.
- 指标、散点图、CSV/Excel 汇总应放在 `Metrics_core.py` 或同类 `Metrics_` 文件。
English: metrics, scatter plot, CSV/Excel should in `Metrics_core.py` or same `Metrics_` file.
- ONNX 选择、dummy input、导出文件命名和追溯 JSON 应放在 `Train_export_onnx.py`。
English: ONNX select, dummy input, exportfile name and JSON should in `Train_export_onnx.py`.
- Tool 文件只能用于检查、诊断或一次性试验，不应成为训练运行依赖。
English: Tool fileonly can use check, or property, not should as trainrun.
- 历史目录只能作为历史备份，不作为当前实现依据。
English: only can as, not as current data.
- 修改 `Train_main.py` 面板前必须先得到用户确认。
English: change `Train_main.py` before must first obtain use confirm confirm.
- 公开版不得重新引入训练后旧样本名后处理链路。
English: public releasemust not new train after old sample name after link path.

## 19. 关键维护记录 / maintain.

- 2026-06-18：GG 将 README 扩充为新手说明书，新增环境准备、数据库检查、dry-run、冒烟测试、正式训练、输出解读、断点续训和常见问题排查。
English: 2026-06-18: GG README as beginnernote, new environment environment, data database check, dry-run, smoke test, formal training, read, resume training and.
- 2026-06-17：GG 将 MFPC-HFNet PCA/归一化先验改为每个 Fold 使用 Train 子集重构，并清理 README 中不需要体现的数据发布说明。
English: 2026-06-17: GG MFPC-HFNet PCA/ first change as each Fold use Train, and clean up README in not need need data note.
- 2026-06-16：重构公开版单文件数据库；新增 `Data_PublicSampleDatabase.py` 和 `Data_RebuildPublicSampleDatabase.py`；训练接口支持 `public_single_npz_v1`。
English: 2026-06-16: public release single filedata database; new `Data_PublicSampleDatabase.py` and `Data_RebuildPublicSampleDatabase.py`; train interface `public_single_npz_v1`.
- 2026-06-16：公开版默认目标切换为 SOC，默认数据库指向 `Training Code` 上一级的 `PublicSoilSampleDatabase`，PCA 先验放入 `ModelAssets/`。
English: 2026-06-16: public releasedefault target as SOC, defaultdata database `Training Code` on `PublicSoilSampleDatabase`, PCA first `ModelAssets/`.
- 2026-06-16：GG 修正公开版 `public_npz` 懒加载样本在训练子集包装中的读取路径，并同步命令行覆盖 `DATASET_ROOT` 时的 `DATA_DIR` 派生逻辑。
English: 2026-06-16: GG public release `public_npz` loadsample in train in readpath, and same line override `DATASET_ROOT` when `DATA_DIR` derivelogic.
- 2026-06-07：ONNX 导出 JSON 增加部署追溯字段，包括数据库格式、输入输出接口、代表 Fold、参数量和 FLOPs_G。
English: 2026-06-07: ONNX export JSON field, database format, inputs and outputs interface, table Fold, count and FLOPs_G.
- 2026-05-30：训练参数优先级收敛为菜单显式设置、`Train_main.py` 面板补充、core 默认；MFPC-HFNet 结构参数迁入菜单层。
English: 2026-05-30: resume training before complete property check; already complete no longer build Dataset.
- 2026-05-29：`Train_core.py` 成为唯一训练引擎；`Menu_*.py` 只保留菜单职责。
English: 2026-05-29: `Train_core.py` as singletraining engine; `Menu_*.py` onlykeep single.
- 2026-05-29：新增 GPU 显存预检、active input 裁剪、模型级 Dataset 复用与训练后自动 ONNX 导出。
English: 2026-05-29: new GPU, active input, Dataset reuse and train after automatically ONNX export.
