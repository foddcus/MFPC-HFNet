# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
训练终端总入口。
Main terminal entry for training.

逻辑说明 / Logic
English: Logic.
----------------
1. 先选择 `Menu_*.py` 菜单接口，再按需覆盖模型选择、训练微调参数、数据库、断点目录和数据集划分默认值。
English: 1. select `Menu_*.py` menu, modelselect, trainingparameter, , directorydefault.
2. 所有终端参数均可为空；未传入、传入空字符串或交互时直接回车，都表示采用代码设计默认值。
English: 2. parameter; , , default.
3. 参数优先级固定为：菜单显式设置 > Train_main 面板补充设置 > 训练代码默认；命令行显式参数用于临时覆盖。
English: 3. parameterPriority: menuexplicit > Train_main > trainingdefault; explicitparameter.
4. 本文件顶部显式保留入口选择、模型选择、断点续训、数据缓存内存容量和训练补充接口；模型结构参数由菜单维护。
English: 4. fileexplicitselect, modelselect, , cachetraining; modelparametermenu.
5. GPU auto-batch/OOM 降批仍由 `Train_support.py` 引擎层管理；这里的容量设置仅指 CPU 数据缓存预算。
English: 5. GPU auto-batch/OOM `Train_support.py` ; CPU cache.
6. 本文件不复制训练循环，只通过 `Train_config.py` 合并参数后调用所选菜单的 `main()`。
English: 6. filetraining, `Train_config.py` parametermenu `main()`.
7. 常规维护中不要随意改动本文件参数面板；MFPC-HFNet 结构超参数不在本入口层设计。
English: 7. fileparameter; MFPC-HFNet parameter.

使用示例 / Examples
English: Examples.
-------------------
1. 按本文件参数面板直接运行：
   python Train_main.py

2. 只查看当前 Full 版本 MFPC-HFNet 最终配置，不启动训练：
   python Train_main.py --dry-run

3. 临时显式指定 MFPC-HFNetV2 菜单，并只查看配置：
   python Train_main.py --menu mfpchfnetv2 --dry-run

最近修改时间 / Last modified: 2026-06-16
English: Last modified: 2026-06-16.
作者 / Author: ljy
English: Author: ljy.
维护记录 / Maintenance:
English: Maintenance:.
- 2026-06-16；作者：GG。默认公开数据库改为工程相对路径，便于公开包整体迁移。
English: - 2026-06-16; Author: GG.defaultpublic databasepath, .
"""

from __future__ import annotations

import os
import sys
from typing import Any

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PUBLIC_DATABASE_ROOT = os.path.normpath(os.path.join(PROJECT_DIR, "..", "PublicSoilSampleDatabase"))

# ================= 0. OpenCV 日志策略 / OpenCV log policy =================
# EN: ================= 0. OpenCV / OpenCV log policy =================.
# 逻辑 / Logic:
# EN: logic / Logic:
# 1. 必须在任何可能间接导入 cv2 的训练模块之前设置，确保 Data_LoaderRuntimeAuto.py 读 TIFF 时生效。
# EN: must in task can can indirectly cv2 train before, confirm Data_LoaderRuntimeAuto.py read TIFF when.
# 2. 部分相机或图像软件写入的 TIFF 私有元数据 tag 65000 会触发 OpenCV warning，但不影响像素矩阵读取。
# EN: or image write TIFF data tag 65000 will OpenCV warning, does not affect image read.
# 3. 这里只在当前训练进程内屏蔽 OpenCV 的 warning/debug/info 日志，ERROR 级别仍保留，避免掩盖真正的图像读取失败。
# EN: hereonly in currenttrain program inside OpenCV warning/debug/info, ERROR still keep, avoid actually imageread.
# 4. 最近修改时间：2026-05-29；作者：ljy。新增入口层 OpenCV TIFF 元数据 warning 屏蔽。
# EN: Last modified: 2026-05-29; Author: ljy. new entry layer OpenCV TIFF data warning.
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from Train_config import (
    DEFAULT_MENU_KEY,
    MENU_REGISTRY,
    SPEC_BY_CLI_NAME,
    load_menu,
    prepare_run,
    summarize_run,
)


# ================= 1. Train_main 显式参数接口 =================
# EN: ================= 1. Train_main form parameters interface =================.
# 逻辑 / Logic:
# EN: logic / Logic:
# 1. 这里是后续最方便手工修改的终端默认参数面板，当前启动入口切到 mfpchfnetv2 的 Full 版本 MFPC-HFNet。
# EN: here is later most manual change terminaldefaultparameters, currentstart interface to mfpchfnetv2 Full MFPC-HFNet.
# 2. 面板保留入口、模型选择、断点续训、数据缓存内存容量和训练补充字段；GPU auto-batch/OOM 不在这里配置。
# EN: keep interface, select, resume training, data cache inside amount and train field; GPU auto-batch/OOM not in here.
# 3. 若手工把某一行改成不同值，且当前菜单没有显式规定该字段，它会作为 Train_main 补充默认生效；命令行传入的非空参数仍可覆盖这里。
# EN: if manual line change not same value, current single not form this field, will as Train_main default; line pass in non- parameters still can overridehere.
# 4. 当前只训练 MFPCHFNetV2_Full，对应 1024x1024 输入的 Full 结构，不训练 H2H3Low / H3Low / LowOnly。
# EN: currentonlytrain MFPCHFNetV2_Full, for should 1024x1024 Full result, does not train H2H3Low / H3Low / LowOnly.
# 5. MFPC-HFNet token / attention / dropout 等结构超参数只保留在 Menu_*.py，不在入口面板设计。
# EN: MFPC-HFNet token / attention / dropout architecture hyperparametersonlykeep in Menu_*.py, not in interface.
# 6. 最近修改时间：2026-06-16；作者：ljy。公开版默认目标变量改为 SOC。
# EN: Last modified: 2026-06-16; Author: ljy.public releasedefaulttarget variable change as SOC.
# 7. 最近修改时间：2026-06-16；作者：GG。公开版默认读取工程同级匿名单文件数据库，便于整包迁移。
# EN: Last modified: 2026-06-16; Author: GG.public releasedefaultread program same anonymized single-file database, to make it easier to migrate.

# ---------- 1.1 启动控制 / Runtime control ----------
# EN: ---------- 1.1 startcontrol / Runtime control ----------.
# 用法示范 / Examples:
# EN: usage examples / Examples:
# - 只核对配置不训练：TRAIN_DRY_RUN = True。
# EN: only for does not train: TRAIN_DRY_RUN = True.
# - 查看菜单或模型清单：TRAIN_LIST_MENUS = True，或 TRAIN_LIST_MODELS = True。
TRAIN_MENU = "mfpchfnetv2"  # 默认菜单：MFPC-HFNet 结构训练；当前只训练 Full 版本。 / EN: default single: MFPC-HFNet train; currentonlytrain Full.
TRAIN_DRY_RUN = False  # 是否只打印最终配置；默认真正启动训练。 / EN: is only; defaultactuallystarttrain.
TRAIN_LIST_MENUS = False  # 是否只列出可选菜单；默认不单独列菜单。 / EN: is only optional single; default single single.
TRAIN_LIST_MODELS = False  # 是否只列出所选菜单模型清单；默认不单独列模型。 / EN: is only single model list; default single.
TRAIN_MODEL_SETTING_KEYS = {"train_model_preset", "train_model_names"}  # 模型选择字段从 Train_main 面板下发。 / EN: selectfield from Train_main below.
TRAIN_RESUME_SETTING_KEYS = {"resume_training", "resume_save_dir"}  # 断点续训字段从 Train_main 面板下发，不写入菜单。 / EN: resume trainingfield from Train_main below, write single.
TRAIN_MEMORY_SETTING_KEYS = {  # CPU 数据缓存容量字段从 Train_main 面板下发；不包含 GPU auto-batch/OOM。 / EN: CPU data cache field from Train_main below; GPU auto-batch/OOM.
    "cache_mode",
    "memory_limit",
    "memory_utilization_ratio",
    "memory_estimate_safety_factor",
    "cache_root",
    "disk_cache_policy",
    "rebuild_preprocess_cache",
    "cache_registry_enabled",
    "cache_registry_filename",
}
TRAIN_DATA_SETTING_KEYS = {  # 数据路径、目标变量和共享折分字段；菜单未显式规定时由 Train_main 补充。 / EN: datapath, target variable and shared foldsfield; single when by Train_main.
    "target_mode",
    "dataset_root",
    "data_dir",
    "gt_path",
    "tn_path",
    "pca_priors_path",
    "full_reference_run_dir",
    "shared_folds_csv_path",
    "load_shared_folds_from_csv",
    "base_run_dir",
    "model_data_dir",
}
TRAIN_TRAINING_SETTING_KEYS = {  # 通用训练控制字段；菜单显式规定时以菜单为准。 / EN: generaltraincontrolfield; single when single as.
    "cleanup_completed_fold_checkpoints",
    "keep_completed_fold_best_model",
    "max_epochs",
    "learning_rate",
    "weight_decay",
    "val_interval",
    "num_folds",
    "num_runs",
    "validation_fold_offset",
    "split_seed",
}
TRAIN_PATIENCE_SETTING_KEYS = {  # LR patience、LR 衰减和小 batch 保护字段。 / EN: LR patience, LR and small batch field.
    "lr_patience_cycles",
    "max_lr_decays",
    "lr_decay_factor",
    "train_loss_patience_bonus_enabled",
    "train_loss_patience_bonus_cycles",
    "train_loss_patience_max_multiplier",
    "lr_patience_grad_accum_extra_multiplier",
    "min_effective_update_batch_size",
    "lr_patience_small_batch_threshold",
    "lr_patience_small_batch_multiplier",
    "lr_patience_batch_one_multiplier",
    "freeze_batchnorm_when_batch_lt_min_effective",
}
TRAIN_OUTPUT_SETTING_KEYS = {  # 输出小数位数、训练后 ONNX 导出等非模型结构字段。 / EN: decimal places, train after ONNX export model architecturefield.
    "export_decimals",
    "export_onnx_after_training",
    "onnx_export_opset",
    "onnx_export_select_fold_by",
    "onnx_export_dynamic_batch",
    "onnx_export_name_template",
}
TRAIN_MODEL_CONFIG_SETTING_KEYS = {  # 模型输入维度字段；共享融合头和结构超参数由菜单维护。 / EN: input dimensionfield; fusion and architecture hyperparameters by single maintain.
    "nir_dim",
    "hyper_dim",
    "image_channels",
}
TRAIN_MAIN_OWNED_SETTING_KEYS = TRAIN_MODEL_SETTING_KEYS | TRAIN_RESUME_SETTING_KEYS | TRAIN_MEMORY_SETTING_KEYS
TRAIN_PANEL_SETTING_KEYS = (
    TRAIN_MAIN_OWNED_SETTING_KEYS
    | TRAIN_DATA_SETTING_KEYS
    | TRAIN_TRAINING_SETTING_KEYS
    | TRAIN_PATIENCE_SETTING_KEYS
    | TRAIN_OUTPUT_SETTING_KEYS
    | TRAIN_MODEL_CONFIG_SETTING_KEYS
)

# ---------- 1.2 菜单与目标变量 / Menu and target ----------
# EN: ---------- 1.2 single and target variable / Menu and target ----------.
# 可选菜单 / Menus:
# EN: optional single / Menus:
# - TRAIN_MENU = "mfpchfnetv2"：论文主模型 MFPC-HFNetV2 Full + 结构消融。
# EN: TRAIN_MENU = "mfpchfnetv2": MFPC-HFNetV2 Full + architecture ablation.
# - TRAIN_MENU = "compare"：AllBackbones baseline 对比。
# EN: TRAIN_MENU = "compare": AllBackbones baseline for.
# - TRAIN_MENU = "input_ablation"：输入端消融，使用 Full 只读参考结果和 input_ablation 专属字段。
# EN: TRAIN_MENU = "input_ablation": input-side ablation, use Full read-only reference result and input_ablation property field.
# preset 用法示范 / Preset examples:
# EN: preset usage examples / Preset examples:
# - mfpchfnetv2: "all" / "full_only" / "ablation_only"。
# - compare: "all" / "classic_only" / "efficientnet_only"。
# - input_ablation: "all" / "single_only" / "pair_only"。
# 手工指定模型示范 / Manual model examples:
# EN: manual / Manual model examples:
# - mfpchfnetv2 的 Full 模型 name："MFPCHFNetV2_Full"。
# EN: mfpchfnetv2 Full name: "MFPCHFNetV2_Full".
# - input_ablation 中的 Full 不进入训练队列，只从 FULL_REFERENCE_RUN_DIR 只读导入，用于结果表对照。
# EN: input_ablation in Full not train column, only from FULL_REFERENCE_RUN_DIR only read, use result result table for.
# - TRAIN_MODEL_NAMES = []：不手工指定，按 TRAIN_MODEL_PRESET 自动展开。
# EN: TRAIN_MODEL_NAMES = []: not manual, by TRAIN_MODEL_PRESET automatically start.
# 目标变量示范 / Target examples:
# EN: target variable / Target examples:
# - TARGET_MODE = "soc"：只训练 SOC；"tn"：只训练 TN；"both"：双目标。
# EN: TARGET_MODE = "soc": onlytrain SOC; "tn": onlytrain TN; "both":.
# 调参影响 / Tuning effect:
# EN: / Tuning effect:
# - TRAIN_MODEL_PRESET 对 mfpchfnetv2 可切换 all / full_only / ablation_only；当前由 TRAIN_MODEL_NAMES 精确锁定 Full 模型。
# EN: TRAIN_MODEL_PRESET for mfpchfnetv2 can all / full_only / ablation_only; current by TRAIN_MODEL_NAMES confirm Full.
# - TRAIN_MODEL_NAMES 指定单个 MFPC-HFNet 结构名时，只训练该结构，不再按 preset 展开其它结构。
# EN: TRAIN_MODEL_NAMES single MFPC-HFNet result name when, onlytrain this result, no longer by preset start its result.
# - TARGET_MODE 从 "soc" 改为 "both"：同时训练 SOC/TN，任务更多且样本会受双真值交集限制。
TRAIN_MODEL_PRESET = "all"  # MFPC-HFNet 结构 preset；当前被 TRAIN_MODEL_NAMES 的单模型清单覆盖。 / EN: MFPC-HFNet preset; current TRAIN_MODEL_NAMES single model listoverride.
TRAIN_MODEL_NAMES = ["MFPCHFNetV2_Full"]  # 只训练 Full 版本 MFPC-HFNet。 / EN: onlytrain Full MFPC-HFNet.
TARGET_MODE = "soc"  # 目标变量模式；公开版标签表只提供 SOC，默认使用 "soc"。 / EN: target-variable mode; public releaselabel only SOC, defaultuse "soc".

# ---------- 1.3 断点续训 / Resume ----------
# EN: ---------- 1.3 resume training / Resume ----------.
# 菜单差异 / Menu-specific defaults:
# EN: single / Menu-specific defaults:
# - mfpchfnetv2 有自己的续训目录与结构消融设置；切回该菜单时需同步检查本面板是否仍适用。
# EN: mfpchfnetv2 and architecture ablation; this single when need same check is no still use.
# - compare 默认 RESUME_TRAINING=True，接续 ModelData\CompareModel。
# EN: compare default RESUME_TRAINING=True, ModelData\CompareModel.
# - input_ablation 菜单自身默认新建时间戳目录；Train_main 可显式下发断点续训设置。
# EN: input_ablation single default new when; Train_main can form below resume training.
# 当前 Full 版本 MFPC-HFNet 默认新建时间戳目录；如需续训，需同步设置 RESUME_TRAINING 和 RESUME_SAVE_DIR。
# EN: current Full MFPC-HFNet default new when; need, need same RESUME_TRAINING and RESUME_SAVE_DIR.
# 调参影响 / Tuning effect:
# EN: / Tuning effect:
# - RESUME_TRAINING=True：继续旧目录中未完成的模型/Fold，适合断点恢复；目录不匹配会导致实验混写或直接报错。
# EN: RESUME_TRAINING=True: old in not yet complete /Fold, suitable forresume from checkpoint; not will write or directlyraise an error.
# - RESUME_TRAINING=False：创建新的输出目录，适合正式新实验；不会复用旧训练进度。
RESUME_TRAINING = False  # 是否断点续训；当前 Full 版本 MFPC-HFNet 默认新建输出目录。 / EN: is resume training; current Full MFPC-HFNet default new output directory.
RESUME_SAVE_DIR = ""  # 断点续训目录示例：ModelData/previous_run；公开版默认不续训。 / EN: resume training: ModelData/previous_run; public releasedefault.
CLEANUP_COMPLETED_FOLD_CHECKPOINTS = True  # 完成 Fold 后是否清理临时 checkpoint；默认清理以节省磁盘。 / EN: complete Fold after is clean uptemporary checkpoint; defaultclean up.
KEEP_COMPLETED_FOLD_BEST_MODEL = True  # 清理临时 checkpoint 时是否保留 best_model.pth；默认保留。 / EN: clean uptemporary checkpoint when is keep best_model.pth; defaultkeep.

# ---------- 1.4 数据库与真值文件 / Dataset and labels ----------
# EN: ---------- 1.4 data database and ground-truth file / Dataset and labels ----------.
# 用法示范 / Examples:
# EN: usage examples / Examples:
# - 只换数据库根目录：优先改 DATASET_ROOT；DATA_DIR / GT_PATH / TN_PATH 会按菜单逻辑派生。
# EN: only data database root directory: prefer change DATASET_ROOT; DATA_DIR / GT_PATH / TN_PATH will by single logicderive.
# - 只换 SOC 或 TN 真值文件：单独改 GT_PATH 或 TN_PATH。
# EN: only SOC or TN ground-truth file: single change GT_PATH or TN_PATH.
# - input_ablation 复用 Full 折分：改 FULL_REFERENCE_RUN_DIR 后，SHARED_FOLDS_CSV_PATH 可自动派生。
# EN: input_ablation reuse Full: change FULL_REFERENCE_RUN_DIR after, SHARED_FOLDS_CSV_PATH can automaticallyderive.
# 菜单专属 / Menu-specific:
# EN: single property / Menu-specific:
# - FULL_REFERENCE_RUN_DIR 仅 input_ablation 使用。
# EN: FULL_REFERENCE_RUN_DIR only input_ablation use.
# - compare 的 PCA_PRIORS_PATH 默认使用公开版 ModelAssets 中的 pca_priors_full.pt。
# EN: compare PCA_PRIORS_PATH defaultusepublic release ModelAssets in pca_priors_full.pt.
# 调参影响 / Tuning effect:
# EN: / Tuning effect:
# - FULL_REFERENCE_RUN_DIR 指向不同 Full 结果：输入消融会导入不同 Full 基线和 shared_folds，影响可比性。
# EN: FULL_REFERENCE_RUN_DIR not same Full result result: will not same Full and shared_folds, comparability.
# - LOAD_SHARED_FOLDS_FROM_CSV=True：严格复用 Full 的 8 折划分，输入消融结果可直接与 Full 对照；False 会按种子重建折分。
DATASET_ROOT = DEFAULT_PUBLIC_DATABASE_ROOT  # 公开版匿名单文件数据库根目录；默认指向 Training Code 上一级的 PublicSoilSampleDatabase。 / EN: public releaseanonymized single-file databaseroot directory; default Training Code on PublicSoilSampleDatabase.
DATA_DIR = os.path.join(DATASET_ROOT, "samples")  # 公开版样本文件目录。 / EN: public releasesample file.
GT_PATH = ""  # 公开版标签已写入每个 .npz 样本文件。 / EN: public releaselabel write each.npz sample file.
TN_PATH = ""  # 公开版当前不发布 TN 标签。 / EN: public releasecurrentdoes not publish TN label.
PCA_PRIORS_PATH = os.path.join(PROJECT_DIR, "ModelAssets", "pca_priors_full.pt")  # MFPC-HFNet PCA 先验路径。 / EN: MFPC-HFNet PCA path.
FULL_REFERENCE_RUN_DIR = os.path.join(PROJECT_DIR, "ModelData", "example_full_reference_run")  # 输入端消融专用：Full 只读参考结果目录示例。 / EN: input-side ablation use: Full read-only reference result.
SHARED_FOLDS_CSV_PATH = os.path.join(FULL_REFERENCE_RUN_DIR, "shared_folds", "fold_assignments.csv")  # 共享 8 折划分 CSV 示例。 / EN: 8 CSV.
LOAD_SHARED_FOLDS_FROM_CSV = False  # 是否强制从 SHARED_FOLDS_CSV_PATH 读取折分；输入端消融默认 True，保证与 Full 基线可比。 / EN: is from SHARED_FOLDS_CSV_PATH read; input-side ablationdefault True, ensure and Full can.

# ---------- 1.5 运行输出目录 / Output roots ----------
BASE_RUN_DIR = PROJECT_DIR  # 运行根目录。 / EN: runroot directory.
MODEL_DATA_DIR = os.path.join(PROJECT_DIR, "ModelData")  # 模型、日志、汇总表输出根目录。 / EN: ,, root directory.

# ---------- 1.6 数据缓存内存容量 / Data-cache memory capacity ----------
# EN: ---------- 1.6 data cache inside amount / Data-cache memory capacity ----------.
# 用法示范 / Examples:
# EN: usage examples / Examples:
# - MEMORY_LIMIT = "256GB"：允许数据缓存最多按 256GB 主机内存预算估计。
# EN: MEMORY_LIMIT = "256GB": allowdata cache most more by 256GB inside estimate.
# - MEMORY_UTILIZATION_RATIO = 0.90：只使用 MEMORY_LIMIT 中的 90% 作为安全预算。
# EN: MEMORY_UTILIZATION_RATIO = 0.90: onlyuse MEMORY_LIMIT in 90% as full.
# - CACHE_MODE = "auto"：由 Data_LoaderRuntimeAuto.py 根据样本估算在 memory/disk 间自动选择。
# EN: CACHE_MODE = "auto": by Data_LoaderRuntimeAuto.py data sample in memory/disk automaticallyselect.
# - CACHE_ROOT = None：磁盘缓存默认由 registry 管理；若要指定专用磁盘缓存根目录，可填绝对路径。
# EN: CACHE_ROOT = None: disk cachedefault by registry; if need use disk cacheroot directory, can for path.
# 调参影响 / Tuning effect:
# EN: / Tuning effect:
# - MEMORY_LIMIT / MEMORY_UTILIZATION_RATIO 调大：更容易使用内存缓存，读取快，但主机内存占用更高。
# EN: MEMORY_LIMIT / MEMORY_UTILIZATION_RATIO large: use inside cache, read, inside use high.
# - MEMORY_LIMIT / MEMORY_UTILIZATION_RATIO 调小：更容易切到磁盘缓存，启动和读取可能变慢，但更稳。
# EN: MEMORY_LIMIT / MEMORY_UTILIZATION_RATIO small: to disk cache, start and read can can,.
# - REBUILD_PREPROCESS_CACHE=True：强制重建预处理缓存，适合数据预处理逻辑变化后使用，平时不要打开。
# EN: REBUILD_PREPROCESS_CACHE=True: preprocess cache, suitable fordata logic after use, when do not start.
# - DISK_CACHE_POLICY="reuse_or_build"：优先复用合法磁盘缓存，缺失或失效时自动重建。
CACHE_MODE = "auto"  # 数据缓存模式；可选 "auto" / "memory" / "disk"。 / EN: data cache; optional "auto" / "memory" / "disk".
MEMORY_LIMIT = "256GB"  # CPU 数据缓存容量上限；用于估算是否可用内存缓存。 / EN: CPU data cache upper bound; use is can use inside cache.
MEMORY_UTILIZATION_RATIO = 0.90  # MEMORY_LIMIT 的可用比例；保留一部分内存给系统和训练进程。 / EN: MEMORY_LIMIT can use; keep inside and train.
MEMORY_ESTIMATE_SAFETY_FACTOR = 1.05  # 数据缓存内存估算安全倍率；越大越保守。 / EN: data cache inside full multiplier; large.
CACHE_ROOT = None  # 磁盘缓存根目录；None 表示使用默认 registry/ModelData 策略。 / EN: disk cacheroot directory; None meansusedefault registry/ModelData.
DISK_CACHE_POLICY = "reuse_or_build"  # 磁盘缓存策略；默认复用合法缓存，缺失时构建。 / EN: disk cache; defaultreuse cache, when build.
REBUILD_PREPROCESS_CACHE = False  # 是否强制重建预处理缓存；正式训练通常保持 False。 / EN: is preprocess cache; formal trainingusually False.
CACHE_REGISTRY_ENABLED = True  # 是否启用磁盘缓存注册表；默认启用以复用合法历史缓存。 / EN: is enable cache registry; defaultenable reuse cache.
CACHE_REGISTRY_FILENAME = "disk_cache_registry.json"  # 磁盘缓存注册表文件名。 / EN: cache registryfile.

# ---------- 1.7 训练基础参数 / Basic training hyperparameters ----------
# EN: ---------- 1.7 train parameters / Basic training hyperparameters ----------.
# 用法示范 / Examples:
# EN: usage examples / Examples:
# - 想快速冒烟测试：MAX_EPOCHS = 2，TRAIN_DRY_RUN = False。
# EN: want smoke test: MAX_EPOCHS = 2, TRAIN_DRY_RUN = False.
# - 想保持现有训练策略：不改这些字段，转换函数等效于采用各菜单自己的 Config 默认值。
# EN: want train: not change field, function use each single Config default value.
# - compare 菜单的 WEIGHT_DECAY 与 mfpchfnetv2 不同；只切换菜单时不要手改，系统会自动取 compare 默认值。
# EN: compare single WEIGHT_DECAY and mfpchfnetv2 not same; only single when do not change, will automatically compare default value.
# 调参影响 / Tuning effect:
# EN: / Tuning effect:
# - MAX_EPOCHS 调大：给弱输入组合更多收敛机会，但训练时间和过拟合风险增加；调小：适合冒烟测试，但可能欠拟合。
# EN: MAX_EPOCHS large: more will, train when and fit; small: suitable forsmoke test, can can fit.
# - LEARNING_RATE 调大：收敛可能更快，但震荡/发散风险更高；调小：更稳但更慢，可能需要更多 epoch。
# EN: LEARNING_RATE large: can can, / high; small:, can can need need more epoch.
# - WEIGHT_DECAY 调大：正则更强、过拟合风险下降，但可能欠拟合；调小：拟合能力更强，但弱输入组合更易过拟合。
# EN: WEIGHT_DECAY large: then, fit below, can can fit; small: fit can, fit.
# - VAL_INTERVAL 调大：验证更少、训练略快，但 best checkpoint 和 LR 衰减触发更迟；调小：监控更密，验证开销增加。
MAX_EPOCHS = 1000  # 输入端消融最大训练 epoch 数；runtime Config 当前默认 1600。 / EN: input-side ablationmaximumtrain epoch; runtime Config currentdefault 1600.
LEARNING_RATE = 1e-4  # 输入端消融基础学习率；与当前主模型默认 LR=1e-4 对齐。 / EN: input-side ablation learning rate; and current default LR=1e-4 alignment.
WEIGHT_DECAY = 1e-3  # 输入端消融 AdamW 权重衰减；runtime Config 当前默认 1e-3。 / EN: input-side ablation AdamW weight decay; runtime Config currentdefault 1e-3.
VAL_INTERVAL = 1  # 验证间隔；当前工程通常用 1，表示每个 epoch 都验证。 / EN: validate; current usually use 1, means each epoch validate.

# ---------- 1.8 数据集划分 / Cross-validation split ----------
# EN: ---------- 1.8 data / Cross-validation split ----------.
# 当前输入消融论文设置 / Current input-ablation setting: 8 folds, run all 8 folds.
# EN: current / Current input-ablation setting: 8 folds, run all 8 folds.
# 调参影响 / Tuning effect:
# EN: / Tuning effect:
# - NUM_FOLDS / NUM_RUNS 调大：统计覆盖更完整，但训练时间近似线性增加；调小：更快，但结果不再是完整 8 折。
# EN: NUM_FOLDS / NUM_RUNS large: override complete, train when property; small:, result result no longer is complete 8.
# - SPLIT_SEED 改变：会产生新的折分，不能再与旧 Full/shared_folds 直接一一对比。
NUM_FOLDS = 8  # 交叉验证总折数；当前工程论文结果使用 8 折。 / EN: cross-validation; current use 8.
NUM_RUNS = 8  # 实际运行 Fold 数；正式输入消融默认完整 8 折，调试可临时改 1。 / EN: run Fold; formal defaultcomplete 8, can temporary change 1.
VALIDATION_FOLD_OFFSET = 1  # 验证折相对测试折的偏移；通常保持 1，避免改变既有折分语义。 / EN: validate for test; usually 1, avoid change.
SPLIT_SEED = 20260317  # 稳定折分随机种子。与 Full 对照时保持不变；需要全新折分时才修改。 / EN: stable split random seed. and Full for when; need full new when change.

# ---------- 1.9 LR patience 与小 batch 策略 / LR patience and small batch ----------
# EN: ---------- 1.9 LR patience and small batch / LR patience and small batch ----------.
# 调参影响 / Tuning effect:
# EN: / Tuning effect:
# - LR_PATIENCE_CYCLES 调大：更晚降低学习率，适合慢收敛弱输入组合；调小：更快降 LR，但可能过早压低学习率。
# EN: LR_PATIENCE_CYCLES large: low learning rate, suitable for; small: LR, can can low learning rate.
# - MAX_LR_DECAYS 调大：允许更多次降 LR 和更长训练；调小：更早停止无改进训练。
# EN: MAX_LR_DECAYS large: allow more LR and train; small: no change train.
# - LR_DECAY_FACTOR 调大到接近 1：每次降 LR 更温和；调小到更低：学习率骤降更强，可能更快稳定也可能过早停滞。
# EN: LR_DECAY_FACTOR large to 1: each LR and; small to low: learning rate, can can stable also can can.
# - TRAIN_LOSS_PATIENCE_* 调大：训练 loss 仍改善时更耐心；调小：更依赖验证集，可能更早降 LR/停止。
# EN: TRAIN_LOSS_PATIENCE_* large: train loss still change when; small: validation set, can can LR/.
# - MIN_EFFECTIVE_UPDATE_BATCH_SIZE 调大：小 batch 时梯度累积步数增加，更新更平滑但单次有效更新更慢；调小则相反。
LR_PATIENCE_CYCLES = 15  # 基础 LR patience，以验证次数为单位；输入端消融当前默认 20。 / EN: LR patience, validate as unit; input-side ablationcurrentdefault 20.
MAX_LR_DECAYS = 2  # 最大学习率衰减次数； / EN: maximumlearning rate;
LR_DECAY_FACTOR = 0.1  # 每次触发 LR 衰减时的倍率；当前默认降到原来的 10%。 / EN: each LR when multiplier; currentdefault to 10%.
TRAIN_LOSS_PATIENCE_BONUS_ENABLED = True  # 训练损失改善时是否临时延长 patience；当前默认启用。 / EN: train change when is temporary patience; currentdefaultenable.
TRAIN_LOSS_PATIENCE_BONUS_CYCLES = 3  # 训练损失改善时增加的 patience 单位；输入端消融等于 LR_PATIENCE_CYCLES。 / EN: train change when patience unit; input-side ablation LR_PATIENCE_CYCLES.
TRAIN_LOSS_PATIENCE_MAX_MULTIPLIER = 2 # 训练损失临时 patience 上限倍率； / EN: train temporary patience upper boundmultiplier;
LR_PATIENCE_GRAD_ACCUM_EXTRA_MULTIPLIER = 1.5  # 低 batch 梯度累积时 patience 额外倍率；当前默认 1.5。 / EN: low batch when patience multiplier; currentdefault 1.5.
MIN_EFFECTIVE_UPDATE_BATCH_SIZE = 8  # 梯度累积目标有效 batch 下限；当前工程默认 8。 / EN: batch lower bound; current default 8.
LR_PATIENCE_SMALL_BATCH_THRESHOLD = 8  # 旧小 batch patience 阈值兼容字段。 / EN: old small batch patience thresholdcompatiblefield.
LR_PATIENCE_SMALL_BATCH_MULTIPLIER = 2  # 旧小 batch patience 倍率兼容字段。 / EN: old small batch patience multipliercompatiblefield.
LR_PATIENCE_BATCH_ONE_MULTIPLIER = 8  # 旧 batch=1 patience 倍率兼容字段。 / EN: old batch=1 patience multipliercompatiblefield.
FREEZE_BATCHNORM_WHEN_BATCH_LT_MIN_EFFECTIVE = True  # 小 batch 时是否冻结 BatchNorm running statistics。 / EN: small batch when is BatchNorm running statistics.

# ---------- 1.10 输入维度与导出 / Input dimensions and export ----------
# EN: ---------- 1.10 input dimension and export / Input dimensions and export ----------.
# 用法示范 / Examples:
# EN: usage examples / Examples:
# - EXPORT_DECIMALS = 3：论文表格和预测 CSV 保留 3 位小数。
# EN: EXPORT_DECIMALS = 3: table and CSV keep 3 small number.
# - EXPORT_ONNX_AFTER_TRAINING = True：每个模型全部 Fold 完成后自动导出 1 个代表 ONNX。
# EN: EXPORT_ONNX_AFTER_TRAINING = True: each full Fold complete after automaticallyexport 1 table ONNX.
# - ONNX_EXPORT_NAME_TEMPLATE 支持 {model_name}/{display_name}/{target_mode}/{fold}/{metric}。
# EN: ONNX_EXPORT_NAME_TEMPLATE {model_name}/{display_name}/{target_mode}/{fold}/{metric}.
# - NIR_DIM / HYPER_DIM / IMAGE_CHANNELS 必须与 Data_LoaderRuntimeAuto.py 输出一致，通常不建议改。
# EN: NIR_DIM / HYPER_DIM / IMAGE_CHANNELS must and Data_LoaderRuntimeAuto.py consistent, usually not recommended change.
# 调参影响 / Tuning effect:
# EN: / Tuning effect:
# - EXPORT_DECIMALS 调大：保留更多小数，文件更细但论文表更冗长；调小：表格更简洁但损失精度。
# EN: EXPORT_DECIMALS large: keep more small number, file table; small: table degree.
# - ONNX_EXPORT_SELECT_FOLD_BY = "test_rmse"：按测试集 RMSE 选择部署代表 Fold，会写入 export_info.json 追溯。
# EN: ONNX_EXPORT_SELECT_FOLD_BY = "test_rmse": by test set RMSE select table Fold, will write export_info.json.
# - ONNX_EXPORT_NAME_TEMPLATE 改成固定文件名时，多模型菜单可能重名；当前只训练单模型，可使用固定部署名。
# EN: ONNX_EXPORT_NAME_TEMPLATE change fixedfile name when, more single can can name; currentonlytrain single, can usefixed name.
# - 最近修改时间：2026-06-07；作者：ljy。将当前单模型 ONNX 文件名固定为 DLV-3-Both-Full。
EXPORT_DECIMALS = 3  # CSV 与汇总表导出小数位数。 / EN: CSV and exportdecimal places.
EXPORT_ONNX_AFTER_TRAINING = True  # 训练完成后是否自动导出每模型代表 ONNX。 / EN: traincomplete after is automaticallyexport each ONNX.
ONNX_EXPORT_OPSET = 18  # ONNX 导出 opset；沿用应用端兼容的历史设置 18。 / EN: ONNX export opset; use should use compatible 18.
ONNX_EXPORT_SELECT_FOLD_BY = "test_rmse"  # 代表 Fold 选择规则；当前支持 test_rmse。 / EN: Fold selectrule; current test_rmse.
ONNX_EXPORT_DYNAMIC_BATCH = True  # ONNX 输入输出第 0 维是否使用动态 batch。 / EN: ONNX inputs and outputs 0 is use batch.
ONNX_EXPORT_NAME_TEMPLATE = "DLV-3-Both-Full.onnx"  # 当前单模型部署 ONNX 文件名。 / EN: current single ONNX file.
NIR_DIM = 5  # NIR 输入维度。 / EN: NIR input dimension.
HYPER_DIM = 681  # HyperVISNIR 输入维度。 / EN: HyperVISNIR input dimension.
IMAGE_CHANNELS = 8  # 多光谱图像通道数。 / EN: more image.
# ---------- 1.11 高级兜底 Config 覆盖 / Advanced fallback Config overrides ----------
# EN: ---------- 1.11 high Config override / Advanced fallback Config overrides ----------.
# 用法示范 / Examples:
# EN: usage examples / Examples:
# - 当字段没有在上方面板中显式列出，但 legacy Config 中确实存在时，可用 ATTR=VALUE 临时覆盖。
# EN: field not in on in form column, legacy Config in confirm in when, can use ATTR=VALUE temporary override.
# - 这里不会新增不存在的 Config 字段；拼写错误会直接报错，避免静默失效。
# EN: here not will new does not exist Config field; write error will directlyraise an error, avoid.
# TRAIN_SET_CONFIG_OVERRIDES = ["EXPORT_DECIMALS=4", "FREEZE_BATCHNORM_WHEN_BATCH_LT_MIN_EFFECTIVE=true"]
TRAIN_SET_CONFIG_OVERRIDES: list[str] = []


_MISSING = object()


def _is_blank_value(value: Any) -> bool:
    """
    判断 Train_main 面板中的空值。
    English: determine Train_main empty value.

    最近修改时间 / Last modified: 2026-05-24
    English: Last modified: 2026-05-24.
    作者 / Author: ljy
    English: Author: ljy.
    """

    return value is None or (isinstance(value, str) and value.strip() == "")


def _resolve_requested_menu_key(argv: list[str] | None) -> str:
    """
    预判本次使用的菜单，用于判断目标菜单显式字段和支持字段。
    English: menu, determinemenuexplicitfieldfield.

    说明：
    English: :
    - 这里只解析 --menu / --menu=xxx，不替代正式 argparse；
    English: --menu=xxx，不替代正式 argparse；.
    - 若命令行未指定菜单，则使用 Train_main 面板中的 TRAIN_MENU；
    English: - menu, Train_main TRAIN_MENU;
    - 若面板也为空，则回退到 Train_config.py 的 DEFAULT_MENU_KEY。
    English: - , fall back Train_config.py DEFAULT_MENU_KEY.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    argv = [] if argv is None else list(argv)
    for index, item in enumerate(argv):
        if item == "--menu" and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--menu="):
            return item.split("=", 1)[1]

    if _is_blank_value(TRAIN_MENU):
        return DEFAULT_MENU_KEY
    return str(TRAIN_MENU).strip()


def _get_menu_code_default(menu_key: str, cli_name: str) -> Any:
    """
    从目标菜单 Config 中读取某个终端字段当前代码默认值。
    English: menu Config readfieldcurrentdefault.

    返回 _MISSING 表示当前菜单不支持该字段；调用方据此避免把 compare /
    English: return _MISSING currentmenufield; avoid compare /.
    input_ablation 专属字段误传给其它菜单。
    English: input_ablation fieldmenu.

    最近修改时间 / Last modified: 2026-05-24
    English: Last modified: 2026-05-24.
    作者 / Author: ljy
    English: Author: ljy.
    """

    if menu_key not in MENU_REGISTRY:
        return _MISSING
    spec = SPEC_BY_CLI_NAME.get(cli_name)
    if spec is None:
        return _MISSING
    if spec.menu_keys and menu_key not in spec.menu_keys:
        return _MISSING

    menu = load_menu(menu_key)
    if not hasattr(menu.Config, spec.attr_name):
        return _MISSING
    return getattr(menu.Config, spec.attr_name)


def _menu_declares_config_field(menu_key: str, cli_name: str) -> bool:
    """
    判断目标菜单是否显式规定某个 Config 字段。
    English: determinemenuexplicit Config field.

    说明:
    English: :
    - 只检查菜单 Config.__dict__，继承自 CommonTrainConfig 的字段不算菜单显式规定；
    English: - checkmenu Config.__dict__, CommonTrainConfig fieldmenuexplicit;
    - 该判断只用于 Train_main 面板补充层，不影响命令行显式覆盖。
    English: - determine Train_main , explicit.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    spec = SPEC_BY_CLI_NAME.get(cli_name)
    if spec is None or menu_key not in MENU_REGISTRY:
        return False
    if spec.menu_keys and menu_key not in spec.menu_keys:
        return False
    menu = load_menu(menu_key)
    return spec.attr_name in getattr(menu.Config, "__dict__", {})


def _menu_aware_design_value(menu_key: str, cli_name: str, panel_value: Any, *, allow_menu_declared: bool = False) -> Any:
    """
    将“写在面板里的默认值”转换为真正需要下发的设计覆盖值。
    English: “default”.

    规则：
    English: :
    - 若目标菜单显式规定该字段，且该字段不是 Train_main 自有入口字段，返回 None；
    English: - menuexplicitfield, field Train_main field, return None;
    - 若目标菜单不支持该字段，返回 None，避免 compare/input_ablation 专属字段互相干扰；
    English: - menufield, return None, avoid compare/input_ablation field;
    - 只要菜单未显式规定且目标菜单支持该字段，就把 Train_main 面板值作为补充设置下发。
    English: - menuexplicitmenufield, Train_main .

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    if not bool(allow_menu_declared) and _menu_declares_config_field(menu_key, cli_name):
        return None

    target_menu_value = _get_menu_code_default(menu_key, cli_name)
    if target_menu_value is _MISSING:
        return None
    return panel_value


def build_train_main_design_defaults(argv: list[str] | None = None) -> dict[str, Any]:
    """
    将 Train_main.py 顶部显式参数面板转换为终端参数层可读取的字典。
    English: Train_main.py explicitparameterparameterreaddictionary.

    物理/工程意义：
    English: /:
    - 顶部变量保留给用户后续直接修改；
    English: - ;
    - 面板中显式写出的默认值是 Train_main 入口补充层；
    English: - explicitdefault Train_main ;
    - 菜单显式声明的 Config 字段不被面板覆盖；
    English: - menuexplicit Config field;
    - 菜单未显式声明但目标菜单支持的 Config 字段由面板补充下发。
    English: - menuexplicitmenu Config field.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    menu_key = _resolve_requested_menu_key(argv)
    panel_defaults = {
        "menu": TRAIN_MENU,
        "interactive": False,
        "prompt_all": False,
        "yes": True,
        "dry_run": TRAIN_DRY_RUN,
        "list_menus": TRAIN_LIST_MENUS,
        "list_models": TRAIN_LIST_MODELS,
        "train_model_preset": TRAIN_MODEL_PRESET,
        "train_model_names": TRAIN_MODEL_NAMES,
        "target_mode": TARGET_MODE,
        "resume_training": RESUME_TRAINING,
        "resume_save_dir": RESUME_SAVE_DIR,
        "cleanup_completed_fold_checkpoints": CLEANUP_COMPLETED_FOLD_CHECKPOINTS,
        "keep_completed_fold_best_model": KEEP_COMPLETED_FOLD_BEST_MODEL,
        "dataset_root": DATASET_ROOT,
        "data_dir": DATA_DIR,
        "gt_path": GT_PATH,
        "tn_path": TN_PATH,
        "pca_priors_path": PCA_PRIORS_PATH,
        "full_reference_run_dir": FULL_REFERENCE_RUN_DIR,
        "shared_folds_csv_path": SHARED_FOLDS_CSV_PATH,
        "load_shared_folds_from_csv": LOAD_SHARED_FOLDS_FROM_CSV,
        "base_run_dir": BASE_RUN_DIR,
        "model_data_dir": MODEL_DATA_DIR,
        "cache_mode": CACHE_MODE,
        "memory_limit": MEMORY_LIMIT,
        "memory_utilization_ratio": MEMORY_UTILIZATION_RATIO,
        "memory_estimate_safety_factor": MEMORY_ESTIMATE_SAFETY_FACTOR,
        "cache_root": CACHE_ROOT,
        "disk_cache_policy": DISK_CACHE_POLICY,
        "rebuild_preprocess_cache": REBUILD_PREPROCESS_CACHE,
        "cache_registry_enabled": CACHE_REGISTRY_ENABLED,
        "cache_registry_filename": CACHE_REGISTRY_FILENAME,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "val_interval": VAL_INTERVAL,
        "num_folds": NUM_FOLDS,
        "num_runs": NUM_RUNS,
        "validation_fold_offset": VALIDATION_FOLD_OFFSET,
        "split_seed": SPLIT_SEED,
        "lr_patience_cycles": LR_PATIENCE_CYCLES,
        "max_lr_decays": MAX_LR_DECAYS,
        "lr_decay_factor": LR_DECAY_FACTOR,
        "train_loss_patience_bonus_enabled": TRAIN_LOSS_PATIENCE_BONUS_ENABLED,
        "train_loss_patience_bonus_cycles": TRAIN_LOSS_PATIENCE_BONUS_CYCLES,
        "train_loss_patience_max_multiplier": TRAIN_LOSS_PATIENCE_MAX_MULTIPLIER,
        "lr_patience_grad_accum_extra_multiplier": LR_PATIENCE_GRAD_ACCUM_EXTRA_MULTIPLIER,
        "min_effective_update_batch_size": MIN_EFFECTIVE_UPDATE_BATCH_SIZE,
        "lr_patience_small_batch_threshold": LR_PATIENCE_SMALL_BATCH_THRESHOLD,
        "lr_patience_small_batch_multiplier": LR_PATIENCE_SMALL_BATCH_MULTIPLIER,
        "lr_patience_batch_one_multiplier": LR_PATIENCE_BATCH_ONE_MULTIPLIER,
        "freeze_batchnorm_when_batch_lt_min_effective": FREEZE_BATCHNORM_WHEN_BATCH_LT_MIN_EFFECTIVE,
        "export_decimals": EXPORT_DECIMALS,
        "export_onnx_after_training": EXPORT_ONNX_AFTER_TRAINING,
        "onnx_export_opset": ONNX_EXPORT_OPSET,
        "onnx_export_select_fold_by": ONNX_EXPORT_SELECT_FOLD_BY,
        "onnx_export_dynamic_batch": ONNX_EXPORT_DYNAMIC_BATCH,
        "onnx_export_name_template": ONNX_EXPORT_NAME_TEMPLATE,
        "nir_dim": NIR_DIM,
        "hyper_dim": HYPER_DIM,
        "image_channels": IMAGE_CHANNELS,
        "set_config": list(TRAIN_SET_CONFIG_OVERRIDES),
    }
    for cli_name in SPEC_BY_CLI_NAME:
        if cli_name in panel_defaults:
            if cli_name not in TRAIN_PANEL_SETTING_KEYS:
                panel_defaults[cli_name] = None
                continue
            panel_defaults[cli_name] = _menu_aware_design_value(
                menu_key,
                cli_name,
                panel_defaults[cli_name],
                allow_menu_declared=cli_name in TRAIN_MAIN_OWNED_SETTING_KEYS,
            )
    return panel_defaults
def main(argv: list[str] | None = None) -> int:
    """
    解析终端参数并启动训练。
    English: parseparametertraining.

    最近修改时间 / Last modified: 2026-05-27
    English: Last modified: 2026-05-27.
    作者 / Author: ljy
    English: Author: ljy.
    """

    argv = list(sys.argv[1:] if argv is None else argv)
    prepared = prepare_run(argv, design_defaults=build_train_main_design_defaults(argv))
    print(summarize_run(prepared))

    if prepared.dry_run:
        print("\n[dry-run] 仅打印配置，不启动训练。")
        return 0

    print("\n开始训练...")
    prepared.menu.run()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


