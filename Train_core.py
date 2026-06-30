# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
共用训练引擎。
Shared training engine.

逻辑说明 / Logic
English: Logic.
----------------
1. 本文件是 V2 唯一“厨师”层训练引擎，只接收 `Menu_*.py` 菜单传入的模型清单和训练微调设置。
English: 1. file V2 single“”training engine, `Menu_*.py` menumodeltraining.
2. 菜单负责“点菜”：声明本次要训练哪些未训练模型、从小到大的执行顺序、模型显示名、输入尺寸、batch size 和特殊参数。
English: 2. menu“”: trainingtrainingmodel, , model, Input, batch size parameter.
3. `Model_*.py` 保存底层网络结构与构建函数；训练引擎只能按菜单 `ModelSpec` 调用模型代码，禁止自行猜测模型或补写隐藏参数。
English: 3. `Model_*.py` savebuild; training enginemenu `ModelSpec` model, modelparameter.
4. 数据读取只属于 `Data_*.py`；指标与结果汇总只属于 `Metrics_*.py`。
English: 4. read `Data_*.py`; metricresult `Metrics_*.py`.
5. Optimizer 策略由 `Train_optimizer.py` 执行，菜单只声明策略，模型只声明参数分组语义。
English: 5. Optimizer `Train_optimizer.py` , menu, modelparameter groups.
6. V2 禁止再拆出多个训练执行主体；训练执行边界统一收敛到 `Train_core.py`。
English: 6. V2 training; training `Train_core.py`.
7. 本轮重构只改变代码组织边界，不主动修改学习率、epoch、weight decay、断点目录等用户训练参数。
English: 7. , learning rate, epoch, weight decay, directorytrainingparameter.
8. LR 衰减训练采用一轮验证策略：第一次降学习率后若仍未刷新验证 best，则结束当前 Fold，不再进入后续降 LR 轮次。
English: 8. LR training: learning rate best, current Fold, LR .

最近修改时间 / Last modified: 2026-06-17
English: Last modified: 2026-06-17.
作者 / Author: ljy / GG
English: Author: ljy / GG.
维护记录 / Maintenance:
English: Maintenance:.
- 2026-06-17；作者：GG。MFPC-HFNet 图像先验改为每个 Fold 仅由 Train 子集重构，
English: - 2026-06-17; Author: GG.MFPC-HFNet image Fold Train ,.
  并在 checkpoint、run_info 和 ONNX 导出路径中记录追溯信息。
  English: checkpoint, run_info ONNX exportpath.
- 2026-06-17；作者：GG。新增同一实验输出根目录内的 Fold PCA 共享缓存，避免图像先验一致的模型重复构建。
English: - 2026-06-17; Author: GG.Outputdirectory Fold PCA cache, avoidimagemodelbuild.
"""

from __future__ import annotations

import json
import os
import shutil
import csv
import sys
import hashlib
from datetime import datetime
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingConfig:
    """
    单个菜单入口的说明信息。
    Metadata for one menu entry.

最近修改时间 / Last modified: 2026-05-29
English: Last modified: 2026-05-29.
作者 / Author: ljy
English: Author: ljy.
输出维护 / Output maintenance:
English: Output maintenance:.
- 2026-05-29；作者：ljy。训练 epoch 的 loss/val_rmse 状态改为终端动态单行刷新，避免长期训练时输出栏持续增长。
English: - 2026-05-29; Author: ljy.training epoch loss/val_rmse , avoidtrainingOutput.
- 2026-05-29；作者：ljy。动态训练状态行改用紧凑字段，避免最佳验证表现刷新时因文本过长被终端自动折行。
English: - 2026-05-29; Author: ljy.trainingfield, avoid.
- 2026-05-29；作者：ljy。普通训练状态保持单行刷新；刷新最佳验证表现时主动换行，固定保留该次 best 记录。
English: - 2026-05-29; Author: ljy.training; , best .
- 2026-05-29；作者：ljy。接入 Train_optimizer.py 的通用 optimizer 策略执行，恢复 Compare/SwinV2 的冻结预热与分组学习率能力。
English: - 2026-05-29; Author: ljy. Train_optimizer.py general optimizer policy, Compare/SwinV2 learning rate.
"""

    name: str
    active_entrypoint: str
    manifest_filename: str
    summary_prefix: str


@dataclass(frozen=True)
class ModelSpec:
    """
    菜单传给训练引擎的模型条目。
    Model item passed from a menu to the training engine.

    字段说明 / Field notes:
    English: Field notes:.
    - `model_family`: 菜单给出的模型代码路径标记，训练引擎只能按该标记和本条目参数调用模型代码；
    English: - `model_family`: menumodelpath, training engineparametermodel;
    - `structure`: MFPC-HFNet 的结构分支描述；
    English: - `structure`: MFPC-HFNet ;
    - `active_inputs`: 当前模型实际启用的输入源；留空表示使用 image / hyper / nir 全输入；
    English: hyper / nir 全输入；.
    - `resolution`: 对比模型的输入分辨率标签；
    English: - `resolution`: modelInputlabel;
    - `extra`: 少量菜单专属元数据，避免为一次性字段继续扩张接口。
    English: - `extra`: menu, avoidfield.

    最近修改时间 / Last modified: 2026-06-17
    English: Last modified: 2026-06-17.
    作者 / Author: ljy
    English: Author: ljy.
    维护记录 / Maintenance:
    English: Maintenance:.
    - 2026-05-30；作者：ljy。移除 MFPC-HFNet 结构输入尺寸默认值，结构尺寸只能由菜单显式声明。
    English: - 2026-05-30; Author: ljy. MFPC-HFNet Inputdefault, menuexplicit.
    - 2026-06-16；作者：GG。默认公开数据库改为工程相对路径，便于公开包整体迁移。
    English: - 2026-06-16; Author: GG.defaultpublic databasepath, .
    - 2026-06-17；作者：GG。增加 SHARED_PCA_PRIOR_CACHE_ENABLED，控制 Fold PCA 共享缓存复用。
    English: - 2026-06-17; Author: GG. SHARED_PCA_PRIOR_CACHE_ENABLED, Fold PCA cache.
    """

    name: str
    display_name: str
    model_family: str
    batch_size: int = 32
    image_size: tuple[int, int] = (1024, 1024)
    structure: str = ""
    active_inputs: tuple[str, ...] = ()
    priors_path: str = ""
    resolution: str = ""
    backbone_name: str = ""
    optimizer_policy: str = ""
    backbone_lr: Optional[float] = None
    head_lr: Optional[float] = None
    freeze_backbone_epochs: int = 0
    disable_dropout_droppath: bool = False
    extra: dict[str, Any] | None = None


class CommonTrainConfig:
    """
    三份菜单共用的 Config 默认字段。
    Shared Config fields used by the three menus.

    说明 / Notes:
    English: Notes:.
    - `Train_main.py` 和 `Train_config.py` 通过这些字段做显式覆盖；
    English: - `Train_main.py` `Train_config.py` fieldexplicit;
    - 菜单可在自己的 `Config` 子类中覆盖具体默认值；
    English: - menu `Config` default;
    - GPU auto-batch/OOM 策略不放在菜单层，后续由 `Train_core.py` 内部执行。
    English: - GPU auto-batch/OOM menu, `Train_core.py` .
    - 正式训练默认要求 CUDA 可用，避免环境异常时静默退回 CPU。
    English: - trainingdefault CUDA , avoid CPU.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    维护记录 / Maintenance:
    English: Maintenance:.
    - 2026-05-30；作者：ljy。移除 MFPC-HFNet 结构输入尺寸默认值，结构尺寸只能由菜单显式声明。
    English: - 2026-05-30; Author: ljy. MFPC-HFNet Inputdefault, menuexplicit.
    - 2026-06-16；作者：GG。默认公开数据库改为工程相对路径，便于公开包整体迁移。
    English: - 2026-06-16; Author: GG.defaultpublic databasepath, .
    """

    SCRIPT_DIR = str(PROJECT_DIR)
    BASE_RUN_DIR = str(PROJECT_DIR)
    MODEL_DATA_DIR = str(PROJECT_DIR / "ModelData")

    TARGET_MODE = "soc"
    TRAIN_MODEL_PRESET = "all"
    TRAIN_MODEL_NAMES: list[str] = []

    DATASET_ROOT = os.path.normpath(str(PROJECT_DIR.parent / "PublicSoilSampleDatabase"))
    DATA_DIR = os.path.join(DATASET_ROOT, "samples")
    GT_PATH = ""
    TN_PATH = ""
    PCA_PRIORS_PATH = os.path.join(SCRIPT_DIR, "ModelAssets", "pca_priors_full.pt")

    RESUME_TRAINING = False
    RESUME_SAVE_DIR: Optional[str] = None
    CLEANUP_COMPLETED_FOLD_CHECKPOINTS = True
    KEEP_COMPLETED_FOLD_BEST_MODEL = True

    CACHE_MODE = "auto"
    MEMORY_LIMIT = "128GB"
    MEMORY_UTILIZATION_RATIO = 0.90
    MEMORY_ESTIMATE_SAFETY_FACTOR = 1.05
    CACHE_ROOT = None
    DISK_CACHE_POLICY = "reuse_or_build"
    REBUILD_PREPROCESS_CACHE = False
    CACHE_REGISTRY_ENABLED = True
    CACHE_REGISTRY_FILENAME = "disk_cache_registry.json"
    SHARED_PCA_PRIOR_CACHE_ENABLED = True  # 是否允许同一实验输出根目录内跨模型复用 Fold 级 PCA 先验。 / EN: is allow same root directory inside across modelsreuse Fold PCA.
    REQUIRE_CUDA_FOR_TRAINING = True  # 正式训练要求使用 CUDA GPU；仅 CPU 冒烟测试时才在菜单 Config 中显式改 False。 / EN: formal training use CUDA GPU; only CPU smoke test when in single Config in change False.

    GPU_BATCH_MEMORY_TARGET_UTILIZATION = 0.85
    GPU_BATCH_MEMORY_MIN_BATCH_SIZE = 1
    MAX_EPOCHS = 1000
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    VAL_INTERVAL = 1
    NUM_FOLDS = 8
    NUM_RUNS = 8
    VALIDATION_FOLD_OFFSET = 1
    SPLIT_SEED = 20260317

    LR_PATIENCE_CYCLES = 25
    MAX_LR_DECAYS = 3
    LR_DECAY_FACTOR = 0.5
    TRAIN_LOSS_PATIENCE_BONUS_ENABLED = True
    TRAIN_LOSS_PATIENCE_BONUS_CYCLES = 25
    TRAIN_LOSS_PATIENCE_MAX_MULTIPLIER = 4.0
    LR_PATIENCE_GRAD_ACCUM_EXTRA_MULTIPLIER = 1.5
    MIN_EFFECTIVE_UPDATE_BATCH_SIZE = 8
    LR_PATIENCE_SMALL_BATCH_THRESHOLD = 8
    LR_PATIENCE_SMALL_BATCH_MULTIPLIER = 2.0
    LR_PATIENCE_BATCH_ONE_MULTIPLIER = 8.0
    FREEZE_BATCHNORM_WHEN_BATCH_LT_MIN_EFFECTIVE = True
    EXPORT_DECIMALS = 6
    EXPORT_ONNX_AFTER_TRAINING = True
    ONNX_EXPORT_OPSET = 18
    ONNX_EXPORT_SELECT_FOLD_BY = "test_rmse"
    ONNX_EXPORT_DYNAMIC_BATCH = True
    ONNX_EXPORT_NAME_TEMPLATE = "{model_name}_{target_mode}.onnx"

    NIR_DIM = 5
    HYPER_DIM = 681
    IMAGE_CHANNELS = 8

    FULL_REFERENCE_RUN_DIR = os.path.join(MODEL_DATA_DIR, "example_full_reference_run")
    SHARED_FOLDS_CSV_PATH = os.path.join(FULL_REFERENCE_RUN_DIR, "shared_folds", "fold_assignments.csv")
    LOAD_SHARED_FOLDS_FROM_CSV = True

    @classmethod
    def get_target_names(cls) -> list[str]:
        """
        返回当前目标变量名称列表。
        English: returncurrentnamelist.
        """

        mode = str(getattr(cls, "TARGET_MODE", "soc")).lower()
        if mode == "both":
            return ["SOC", "TN"]
        if mode == "tn":
            return ["TN"]
        return ["SOC"]


SHARED_MODEL_DIM_CONFIG_FIELDS = (
    ("SHARED_IMAGE_EMBED_DIM", "image_embed_dim", int),
    ("SHARED_HYPER_EMBED_DIM", "hyper_embed_dim", int),
    ("SHARED_NIR_EMBED_DIM", "nir_embed_dim", int),
    ("SHARED_FUSION_HIDDEN_DIM", "fusion_hidden_dim", int),
)


MFPCHF_ARCHITECTURE_CONFIG_FIELDS = (
    ("INPUT_FEATURE_VECTOR_RATIO", "input_feature_vector_ratio", float),
    ("ALLOCATION_SOURCE", "allocation_source", str),
    ("FREEZE_PCA", "freeze_pca", bool),
    ("MFPCHF_TOKEN_COMPRESSION_RATIO", "token_compression_ratio", float),
    ("MFPCHF_TOKEN_DIM_MIN", "token_dim_min", int),
    ("MFPCHF_TOKEN_DIM_ROUND_MULTIPLE", "token_dim_round_multiple", int),
    ("MFPCHF_LD_ATTN_DIM", "ld_attn_dim", int),
    ("MFPCHF_LD_HEADS", "ld_heads", int),
    ("MFPCHF_CPE_ATTN_DIM", "cpe_attn_dim", int),
    ("MFPCHF_CPE_HEADS", "cpe_heads", int),
    ("MFPCHF_DROPOUT", "dropout", float),
)


def apply_menu_shared_model_dim_kwargs(build_kwargs: dict[str, Any], config: type[CommonTrainConfig]) -> None:
    """
    将菜单 Config 中显式声明的共享嵌入/融合维度写入模型构造参数。
    English: menu Config explicit/writemodelparameter.

    设计说明:
    English: Design note:
    - 共享嵌入维度会改变模型结构，归菜单层维护；
    English: - model, menu;
    - Train_core 只搬运菜单存在的字段，不在引擎层设计默认结构宽度；
    English: - Train_core menufield, default;
    - 最近修改时间：2026-05-30；作者：ljy。
    English: - Last modified: 2026-05-30; Author: ljy.
    """

    for attr_name, kwarg_name, caster in SHARED_MODEL_DIM_CONFIG_FIELDS:
        if hasattr(config, attr_name):
            build_kwargs[kwarg_name] = caster(getattr(config, attr_name))


def apply_menu_mfpchf_architecture_kwargs(build_kwargs: dict[str, Any], config: type[CommonTrainConfig]) -> None:
    """
    将菜单 Config 中显式声明的 MFPC-HFNet 结构参数写入模型构造参数。
    English: menu Config explicit MFPC-HFNet parameterwritemodelparameter.

    设计说明:
    English: Design note:
    - Train_core 只搬运菜单提供的结构字段，不在引擎层设计 token、attention 或 dropout 默认值；
    English: - Train_core menufield, token, attention dropout default;
    - 若菜单没有声明某个字段，则不写入 build_kwargs，让 Model_MFPCHFNet.py 的构造函数默认值生效；
    English: - menufield, write build_kwargs, Model_MFPCHFNet.py default;
    - 最近修改时间：2026-05-30；作者：ljy。
    English: - Last modified: 2026-05-30; Author: ljy.
    """

    for attr_name, kwarg_name, caster in MFPCHF_ARCHITECTURE_CONFIG_FIELDS:
        if hasattr(config, attr_name):
            build_kwargs[kwarg_name] = caster(getattr(config, attr_name))


class ExperimentMenu:
    """
    菜单与共用训练引擎之间的唯一合同。
    The only contract between a menu and the shared training engine.

    说明 / Notes:
    English: Notes:.
    - `all_model_specs` 的声明顺序就是主管后续执行训练的顺序；
    English: - `all_model_specs` training;
    - 对多个各异模型，应由菜单从小模型到大模型排列，训练引擎不得重新排序或自行增删模型。
    English: - model, menumodelmodel, training enginemodel.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    def __init__(
        self,
        config: TrainingConfig,
        config_cls: type[CommonTrainConfig],
        all_model_specs: Iterable[ModelSpec],
        preset_groups: Optional[dict[str, Iterable[str]]] = None,
    ) -> None:
        """
        保存菜单配置、模型清单和预设组，并同步当前生效模型列表。
        English: savemenuconfiguration, model, currentmodellist.

        输入:
        English: Input:
            config: 训练任务元信息。
            English: config: training.
            config_cls: 菜单暴露的 Config 类。
            English: config_cls: menu Config .
            all_model_specs: 菜单声明的完整模型规格，顺序即训练顺序。
            English: all_model_specs: menumodel, training.
            preset_groups: 可选模型预设组。
            English: preset_groups: optionalmodel.

        最近修改时间 / Last modified: 2026-05-29
        English: Last modified: 2026-05-29.
        作者 / Author: ljy
        English: Author: ljy.
        """
        self.config = config
        self.Config = config_cls
        self.all_model_specs = list(all_model_specs)
        self.ALL_MODEL_SPECS = self.all_model_specs
        self.preset_groups = {key: list(value) for key, value in (preset_groups or {}).items()}
        self.ModelSpec = ModelSpec
        self.MODEL_SPECS: list[ModelSpec] = []
        self.sync_to_engine()

    def select_model_specs(self) -> list[ModelSpec]:
        """
        根据 Config.TRAIN_MODEL_NAMES / TRAIN_MODEL_PRESET 选择本轮模型清单。
        English: TRAIN_MODEL_PRESET 选择本轮模型清单.

        最近修改时间 / Last modified: 2026-05-29
        English: Last modified: 2026-05-29.
        作者 / Author: ljy
        English: Author: ljy.
        """

        requested_names = [str(item).strip() for item in getattr(self.Config, "TRAIN_MODEL_NAMES", []) if str(item).strip()]
        spec_by_name = {spec.name: spec for spec in self.all_model_specs}
        if requested_names:
            missing = [name for name in requested_names if name not in spec_by_name]
            if missing:
                raise ValueError(f"TRAIN_MODEL_NAMES 包含当前菜单不存在的模型: {missing}")
            requested_name_set = set(requested_names)
            return [spec for spec in self.all_model_specs if spec.name in requested_name_set]

        preset = str(getattr(self.Config, "TRAIN_MODEL_PRESET", "all")).strip() or "all"
        if preset == "all":
            return list(self.all_model_specs)
        if preset not in self.preset_groups:
            raise ValueError(f"未知 TRAIN_MODEL_PRESET={preset!r}，可选: {['all', *sorted(self.preset_groups)]}")
        preset_names = self.preset_groups[preset]
        return [spec_by_name[name] for name in preset_names if name in spec_by_name]

    def sync_to_engine(self) -> None:
        """
        同步当前菜单选择结果。
        English: currentmenuselectresult.

        说明：V2 不再把菜单同步到多个训练执行主体；这里只有菜单自身与 Train_core 的合同同步。
        English: : V2 menutraining; menu Train_core .

        最近修改时间 / Last modified: 2026-05-29
        English: Last modified: 2026-05-29.
        作者 / Author: ljy
        English: Author: ljy.
        """

        self.ALL_MODEL_SPECS = self.all_model_specs
        self.MODEL_SPECS = self.select_model_specs()

    def run(self):
        """
        运行当前菜单。
        English: currentmenu.
        """

        return run_training_menu(self)


def serialize_model_spec(spec: ModelSpec) -> dict[str, Any]:
    """
    将 ModelSpec 转为可写入 JSON 的普通字典。
    English: ModelSpec write JSON dictionary.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    payload = asdict(spec)
    payload["image_size"] = list(spec.image_size)
    payload["active_inputs"] = list(spec.active_inputs)
    return payload


def build_training_plan(menu: ExperimentMenu) -> dict[str, Any]:
    """
    构建训练计划摘要。
    English: buildtraining.

    物理意义 / Meaning:
    English: Meaning:.
    - 训练计划是菜单交给厨师前的“点菜单”确认结果；
    English: - trainingmenu“menu”result;
    - 真实训练循环后续只应读取该结构，不再自行解释菜单层默认；
    English: - trainingread, menudefault;
    - `models` 中的 `menu_order` 即主管按菜单从小模型到大模型依次训练的顺序。
    English: - `models` `menu_order` menumodelmodeltraining.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    config = menu.Config
    return {
        "engine": "Train_core",
        "menu": menu.config.name,
        "entrypoint": menu.config.active_entrypoint,
        "target_mode": getattr(config, "TARGET_MODE", None),
        "target_names": config.get_target_names() if hasattr(config, "get_target_names") else [],
        "resume_training": getattr(config, "RESUME_TRAINING", None),
        "resume_save_dir": getattr(config, "RESUME_SAVE_DIR", None),
        "model_data_dir": getattr(config, "MODEL_DATA_DIR", None),
        "num_folds": getattr(config, "NUM_FOLDS", None),
        "num_runs": getattr(config, "NUM_RUNS", None),
        "split_seed": getattr(config, "SPLIT_SEED", None),
        "export_onnx_after_training": getattr(config, "EXPORT_ONNX_AFTER_TRAINING", None),
        "onnx_export_opset": getattr(config, "ONNX_EXPORT_OPSET", None),
        "onnx_export_select_fold_by": getattr(config, "ONNX_EXPORT_SELECT_FOLD_BY", None),
        "onnx_export_dynamic_batch": getattr(config, "ONNX_EXPORT_DYNAMIC_BATCH", None),
        "onnx_export_name_template": getattr(config, "ONNX_EXPORT_NAME_TEMPLATE", None),
        "execution_order_policy": "menu_order_small_to_large",
        "models": [
            {
                "menu_order": index,
                **serialize_model_spec(spec),
            }
            for index, spec in enumerate(menu.MODEL_SPECS, start=1)
        ],
    }


def write_training_plan(menu: ExperimentMenu, plan: dict[str, Any]) -> str:
    """
    将训练计划写入 ModelData，作为 V2 入口边界验证产物。
    English: trainingwrite ModelData, V2 .

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    model_data_dir = Path(getattr(menu.Config, "MODEL_DATA_DIR", PROJECT_DIR / "ModelData"))
    model_data_dir.mkdir(parents=True, exist_ok=True)
    path = model_data_dir / f"{menu.config.summary_prefix}_train_plan.json"
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def resolve_output_root(menu: ExperimentMenu) -> Path:
    """
    解析本次训练输出根目录。
    English: parsetrainingOutputdirectory.

    逻辑说明 / Logic:
    English: Logic:.
    1. 断点续训时优先使用 RESUME_SAVE_DIR，避免把旧实验续到新目录；
    English: 1. RESUME_SAVE_DIR, avoiddirectory;
    2. 新实验按菜单名、目标变量和折数创建时间戳目录；
    English: 2. menu, createdirectory;
    3. 最近修改时间：2026-05-29；作者：ljy。
    English: 3. Last modified: 2026-05-29; Author: ljy.
    """

    config = menu.Config
    if getattr(config, "RESUME_TRAINING", False) and getattr(config, "RESUME_SAVE_DIR", None):
        output_root = Path(str(config.RESUME_SAVE_DIR))
    else:
        target_tag = "_".join(config.get_target_names() if hasattr(config, "get_target_names") else ["SOC"])
        menu_tag = {
            "input_ablation": "InputAblation",
            "mfpchfnetv2": "MFPCHFNetV2_Unified",
            "compare_all_backbones": "CompareModel",
        }.get(menu.config.name, menu.config.summary_prefix)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_root = Path(str(getattr(config, "MODEL_DATA_DIR", PROJECT_DIR / "ModelData"))) / (
            f"{timestamp}_{menu_tag}_{target_tag}_{int(getattr(config, 'NUM_FOLDS', 8))}FoldCV"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def normalize_spec_active_inputs(spec: ModelSpec) -> tuple[str, ...]:
    """
    返回菜单条目的有效输入组合。
    English: returnmenuInput.

    最近修改时间：2026-05-30；作者：ljy。
    English: Last modified: 2026-05-30; Author: ljy.
    """

    if spec.active_inputs:
        return tuple(str(item).strip().lower() for item in spec.active_inputs if str(item).strip())
    return ("image", "hyper", "nir")


def resolve_required_structure_label(spec: ModelSpec, active_inputs: tuple[str, ...] | None = None) -> str:
    """
    返回需要图像结构的模型菜单结构标签。
    English: returnimagemodelmenulabel.

    设计说明：
    English: Design note:
    - MFPC-HFNet 主结构和启用 image 的输入端消融都会实例化图像分支，必须由菜单显式给出结构标签；
    English: - MFPC-HFNet image Inputimage, menuexplicitlabel;
    - 训练引擎只校验和转交菜单值，不再把空结构补写为 Full；
    English: - training enginevalidationmenu, Full;
    - 最近修改时间：2026-05-30；作者：ljy。
    English: - Last modified: 2026-05-30; Author: ljy.
    """

    resolved_inputs = active_inputs if active_inputs is not None else normalize_spec_active_inputs(spec)
    needs_structure = spec.model_family == "mfpchfnet" or (
        spec.model_family == "input_ablation" and "image" in resolved_inputs
    )
    structure_label = str(spec.structure).strip()
    if needs_structure and not structure_label:
        raise ValueError(
            f"{spec.name} 启用了 MFPC-HFNet 图像结构，但菜单 ModelSpec.structure 为空；"
            "请在对应 Menu_*.py 中显式声明结构标签。"
        )
    return structure_label


def format_pyramid_setting_for_spec(spec: ModelSpec, active_inputs: tuple[str, ...] | None = None) -> str:
    """
    返回输出记录中的结构标签。
    English: returnOutputlabel.

    设计说明：
    English: Design note:
    - 输出只写菜单传入的 `ModelSpec.structure`；
    English: - Outputmenu `ModelSpec.structure`;
    - 非图像输入端消融没有图像金字塔，固定写 `none`；
    English: - imageInputimage, `none`;
    - 最近修改时间：2026-05-30；作者：ljy。
    English: - Last modified: 2026-05-30; Author: ljy.
    """

    resolved_inputs = active_inputs if active_inputs is not None else normalize_spec_active_inputs(spec)
    if spec.model_family == "input_ablation" and "image" not in resolved_inputs:
        return "none"
    structure_label = resolve_required_structure_label(spec, resolved_inputs)
    return structure_label if structure_label else "none"


def resolve_model_active_inputs(model: Any, spec: ModelSpec) -> tuple[str, ...]:
    """
    根据模型对象本身解析训练阶段需要搬到 GPU 的输入源。
    English: modelparsetraining GPU Input.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    设计说明：GPU 输入搬运策略优先服从已构建模型的 `active_inputs` 属性，而不是针对某一类消融模型写特例；
    English: Design note: GPU Inputbuildmodel `active_inputs` , model;
    若模型没有声明该属性，则回退到菜单 `ModelSpec.active_inputs`，再回退到 image / hyper / nir 全输入。
    English: hyper / nir 全输入.
    """

    model_active_inputs = getattr(model, "active_inputs", None)
    if model_active_inputs is None and hasattr(model, "module"):
        model_active_inputs = getattr(model.module, "active_inputs", None)
    if model_active_inputs:
        return normalize_active_inputs_for_device(model_active_inputs)
    return normalize_spec_active_inputs(spec)


def get_target_names(config: type[CommonTrainConfig]) -> list[str]:
    """
    返回目标变量名称列表。
    English: returnnamelist.
    """

    if hasattr(config, "get_target_names"):
        return list(config.get_target_names())
    return ["SOC"]


def get_output_dim(config: type[CommonTrainConfig]) -> int:
    """
    返回模型输出维度。
    English: returnmodelOutput.
    """

    return len(get_target_names(config))


def format_size_tag(image_size: Sequence[int]) -> str:
    """
    将图像尺寸转为 1024x1024 形式。
    English: image 1024x1024 .
    """

    return f"{int(image_size[0])}x{int(image_size[1])}"


def format_active_inputs_text(active_inputs: Sequence[str]) -> str:
    """
    将 active_inputs 转为 CSV/JSON 中使用的短标签。
    English: active_inputs CSV/JSON label.
    """

    return "+".join(str(item).strip().lower() for item in active_inputs if str(item).strip())


def format_input_setting(active_inputs: Sequence[str]) -> str:
    """
    将 active_inputs 转为论文结果表中易读的输入组合名。
    English: active_inputs resultInput.
    """

    label_map = {
        "image": "Image",
        "hyper": "HyperVISNIR",
        "nir": "NIR",
    }
    return "+".join(label_map.get(str(item).lower(), str(item)) for item in active_inputs)


def write_terminal_live_status(text: str, previous_width: int = 0) -> int:
    """
    在终端同一行刷新训练状态。
    English: training.

    输入:
    English: Input:
        text: 当前完整状态文本。
        English: text: current.
        previous_width: 上一次状态文本宽度，用于覆盖较长旧内容的尾部。
        English: previous_width: , .
    输出:
    English: Output:
        当前文本宽度，调用方应在下一次刷新时传回。
        English: current, .

    设计说明:
    English: Design note:
        该函数只改变终端显示方式，不参与训练状态保存；validation_history.csv 和
        English: , trainingsave; validation_history.csv.
        training_progress.json 仍按原逻辑逐 epoch 记录完整数据。
        English: training_progress.json Logic epoch .
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
    """
    text = str(text)
    padding = " " * max(int(previous_width) - len(text), 0)
    sys.stdout.write("\r" + text + padding)
    sys.stdout.flush()
    return len(text)


def finish_terminal_live_status(previous_width: int = 0) -> None:
    """
    结束终端动态状态行并换行。
    English: This docstring documents the corresponding function behavior and engineering constraints.

    输入:
    English: Input:
        previous_width: 大于 0 表示此前存在动态状态行。
        English: previous_width: 0 .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    if int(previous_width) > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()


def build_model_from_spec(spec: ModelSpec, config: type[CommonTrainConfig]):
    """
    根据菜单条目调用 Model_*.py 构建未训练模型。
    English: menu Model_*.py buildtrainingmodel.

    注意 / Notes:
    English: Notes:.
    - 特殊输入组合来自菜单 `ModelSpec.active_inputs`；
    English: - Inputmenu `ModelSpec.active_inputs`;
    - 频层结构组合来自菜单 `ModelSpec.structure`，训练引擎只负责转交，不自行推断；
    English: - menu `ModelSpec.structure`, training engine, ;
    - 训练引擎不自行增删模型，不在这里补写菜单没有声明的结构组合；
    English: - training enginemodel, menu;
    - 最近修改时间：2026-05-29；作者：ljy。
    English: - Last modified: 2026-05-29; Author: ljy.
    """

    if spec.model_family == "compare_backbone":
        from Model_CompareBackbones import build_compare_backbone_model

        compare_kwargs = {
            "backbone_name": spec.backbone_name,
            "output_dim": get_output_dim(config),
            "feature_dim_hyper": int(getattr(config, "HYPER_DIM", 681)),
            "feature_dim_nir": int(getattr(config, "NIR_DIM", 5)),
            "image_channels": int(getattr(config, "IMAGE_CHANNELS", 8)),
            "expected_image_hw": tuple(spec.image_size),
            "disable_dropout_droppath": bool(spec.disable_dropout_droppath),
        }
        apply_menu_shared_model_dim_kwargs(compare_kwargs, config)
        return build_compare_backbone_model(**compare_kwargs)

    if spec.model_family not in {"mfpchfnet", "input_ablation"}:
        raise NotImplementedError(
            f"当前 Train_core 真实训练循环暂只接入 MFPC-HFNet / input_ablation / compare_backbone，"
            f"尚未接入 model_family={spec.model_family!r}。"
        )

    from Model_MFPCHFNet import build_model

    extra = dict(spec.extra or {})
    active_inputs = normalize_spec_active_inputs(spec)
    structure_label = resolve_required_structure_label(spec, active_inputs)
    build_kwargs = {
        "pca_priors_path": spec.priors_path or getattr(config, "PCA_PRIORS_PATH", None),
        "output_dim": get_output_dim(config),
        "feature_dim_hyper": int(getattr(config, "HYPER_DIM", 681)),
        "feature_dim_nir": int(getattr(config, "NIR_DIM", 5)),
        "expected_image_hw": tuple(spec.image_size),
        "structure": structure_label,
        "active_inputs": active_inputs,
    }
    apply_menu_shared_model_dim_kwargs(build_kwargs, config)
    apply_menu_mfpchf_architecture_kwargs(build_kwargs, config)
    if "ffn_ratio" in extra:
        build_kwargs["ffn_ratio"] = float(extra["ffn_ratio"])
    return build_model(**build_kwargs)


def build_dataset_for_spec(spec: ModelSpec, config: type[CommonTrainConfig]):
    """
    根据菜单条目构建数据集。
    English: menubuild.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    from Data_LoaderRuntimeAuto import SoilMultiSourceDataset

    data_root = getattr(config, "DATA_DIR", None) or getattr(config, "DATASET_ROOT")
    return SoilMultiSourceDataset(
        data_root=data_root,
        gt_path=getattr(config, "GT_PATH"),
        image_size=tuple(spec.image_size),
        nir_dim=int(getattr(config, "NIR_DIM", 5)),
        target_mode=str(getattr(config, "TARGET_MODE", "soc")),
        tn_path=getattr(config, "TN_PATH", None),
        cache_mode=getattr(config, "CACHE_MODE", "auto"),
        cache_root=getattr(config, "CACHE_ROOT", None),
        rebuild_cache=bool(getattr(config, "REBUILD_PREPROCESS_CACHE", False)),
        memory_limit=getattr(config, "MEMORY_LIMIT", None),
        memory_utilization_ratio=float(getattr(config, "MEMORY_UTILIZATION_RATIO", 0.90)),
        memory_estimate_safety_factor=float(getattr(config, "MEMORY_ESTIMATE_SAFETY_FACTOR", 1.05)),
        disk_cache_policy=getattr(config, "DISK_CACHE_POLICY", "reuse_or_build"),
        cache_registry_enabled=bool(getattr(config, "CACHE_REGISTRY_ENABLED", True)),
        cache_registry_filename=getattr(config, "CACHE_REGISTRY_FILENAME", "disk_cache_registry.json"),
        active_inputs=normalize_spec_active_inputs(spec),
    )


def build_dataset_reuse_key_for_spec(spec: ModelSpec, config: type[CommonTrainConfig]) -> tuple:
    """
    构建模型级 Dataset 复用判定键。
    English: buildmodel Dataset .

    逻辑说明：
    English: Logic notes:
    1. 该键只描述数据读取结果是否完全一致，不描述模型结构、优化器或 batch size；
    English: 1. readresult, model, batch size;
    2. 若连续两个模型的键完全相同，则后一个模型可以复用前一个模型已构建的 Dataset 与 Fold 划分；
    English: 2. model, modelmodelbuild Dataset Fold ;
    3. 若键不同，则说明后续训练所需数据不同，应在当前模型结束后释放旧 Dataset，避免 memory 模式长期占用无用内存。
    English: 3. , training, currentmodel Dataset, avoid memory .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    def normalize_path_value(value: Any) -> str:
        """
        规范化路径字段，便于 Windows 路径比较。
        English: normalizepathfield, Windows path.
        """

        if value is None:
            return ""
        text = str(value).strip()
        return os.path.normcase(os.path.normpath(text)) if text else ""

    data_root = getattr(config, "DATA_DIR", None) or getattr(config, "DATASET_ROOT")
    return (
        normalize_path_value(data_root),
        normalize_path_value(getattr(config, "GT_PATH")),
        normalize_path_value(getattr(config, "TN_PATH", None)),
        tuple(int(value) for value in tuple(spec.image_size)),
        int(getattr(config, "NIR_DIM", 5)),
        str(getattr(config, "TARGET_MODE", "soc")).lower().strip(),
        tuple(normalize_spec_active_inputs(spec)),
        str(getattr(config, "CACHE_MODE", "auto")).lower().strip(),
        normalize_path_value(getattr(config, "CACHE_ROOT", None)),
        bool(getattr(config, "REBUILD_PREPROCESS_CACHE", False)),
        str(getattr(config, "MEMORY_LIMIT", "")),
        float(getattr(config, "MEMORY_UTILIZATION_RATIO", 0.90)),
        float(getattr(config, "MEMORY_ESTIMATE_SAFETY_FACTOR", 1.05)),
        str(getattr(config, "DISK_CACHE_POLICY", "reuse_or_build")).lower().strip(),
        bool(getattr(config, "CACHE_REGISTRY_ENABLED", True)),
        str(getattr(config, "CACHE_REGISTRY_FILENAME", "disk_cache_registry.json")).strip(),
    )


def load_shared_fold_assignments(dataset: Any, config: type[CommonTrainConfig]) -> list[list[int]]:
    """
    从 Full 参考实验的 shared_folds CSV 读取折分。
    English: Full shared_folds CSV read.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    csv_path = Path(str(getattr(config, "SHARED_FOLDS_CSV_PATH")))
    num_folds = int(getattr(config, "NUM_FOLDS", 8))
    folds: list[list[int]] = [[] for _ in range(num_folds)]
    records = getattr(dataset, "data_cache", [])
    seen_indices: set[int] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset_index = int(row["Dataset_Index"])
            fold_id = int(row["Fold"])
            if not (0 <= fold_id < num_folds):
                raise ValueError(f"shared_folds 中 Fold={fold_id} 超出 NUM_FOLDS={num_folds}。")
            if not (0 <= dataset_index < len(records)):
                raise ValueError(f"shared_folds 中 Dataset_Index={dataset_index} 超出当前数据集长度 {len(records)}。")
            sample_name = str(row.get("SampleName", "")).strip()
            current_name = str(records[dataset_index].get("sample_name", "")).strip()
            if sample_name and current_name and sample_name != current_name:
                raise ValueError(
                    f"shared_folds 样本顺序不一致: index={dataset_index}, csv={sample_name}, dataset={current_name}"
                )
            folds[fold_id].append(dataset_index)
            seen_indices.add(dataset_index)

    if len(seen_indices) != len(records):
        raise ValueError(f"shared_folds 样本数 {len(seen_indices)} 与当前数据集样本数 {len(records)} 不一致。")
    return folds


def build_fold_assignments_for_training(dataset: Any, config: type[CommonTrainConfig]) -> list[list[int]]:
    """
    读取或构建训练折分。
    English: readbuildtraining.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    from Train_support import build_stable_fold_assignments

    if bool(getattr(config, "LOAD_SHARED_FOLDS_FROM_CSV", False)) and getattr(config, "SHARED_FOLDS_CSV_PATH", None):
        path = Path(str(getattr(config, "SHARED_FOLDS_CSV_PATH")))
        if path.is_file():
            return load_shared_fold_assignments(dataset, config)
    return build_stable_fold_assignments(dataset, int(getattr(config, "SPLIT_SEED", 20260317)), int(getattr(config, "NUM_FOLDS", 8)))


FOLD_TRAIN_ONLY_PCA_PRIORS_FILENAME = "pca_priors_train_only.pt"
FOLD_TRAIN_ONLY_PCA_PRIORS_SUMMARY_FILENAME = "pca_priors_train_only_summary.json"
FOLD_TRAIN_ONLY_PCA_SHARED_CACHE_DIRNAME = "_shared_pca_priors"
FOLD_TRAIN_ONLY_PCA_SHARED_CACHE_VERSION = "fold_train_only_pca_prior_cache_v1"


def compute_train_indices_sha1(train_indices: Sequence[int]) -> str:
    """
    计算当前 Fold 训练索引的稳定 SHA1。
    English: calculatecurrent Fold training SHA1.

    设计说明:
    English: Design note:
    - 该 hash 用于判断已写出的 train-only 先验是否仍对应该 Fold 的 Train 子集；
    English: - hash determine train-only Fold Train ;
    - 不把 Validation/Test 索引混入 hash，避免误导先验构建范围；
    English: - Validation/Test hash, avoidbuild;
    - 最近修改时间：2026-06-17；作者：GG。
    English: - Last modified: 2026-06-17; Author: GG.
    """

    text = ",".join(str(int(index)) for index in train_indices)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def compute_text_sequence_sha1(values: Sequence[Any]) -> str:
    """
    计算字符串序列的稳定 SHA1。
    English: calculate SHA1.

    设计说明:
    English: Design note:
    - 用换行拼接，避免 ["ab", "c"] 与 ["a", "bc"] 这类边界混淆；
    English: - , avoid ["ab", "c"] ["a", "bc"] ;
    - 主要用于记录当前数据集样本身份和 Train 子集样本身份；
    English: - currentsample Train sample;
    - 最近修改时间：2026-06-17；作者：GG。
    English: - Last modified: 2026-06-17; Author: GG.
    """

    text = "\n".join(str(value) for value in values)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def compute_json_sha1(payload: Any) -> str:
    """
    计算 JSON 可序列化对象的稳定 SHA1。
    English: calculate JSON SHA1.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_trace_path_value(value: Any) -> str:
    """
    规范化写入先验缓存身份的路径字段。
    English: normalizewritecachepathfield.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    if value in (None, ""):
        return ""
    return os.path.normcase(os.path.normpath(str(value)))


def get_fold_pca_prior_builder_parameter_trace() -> dict[str, Any]:
    """
    记录 Fold 级 PCA 先验构建参数。
    English: Fold PCA buildparameter.

    设计说明:
    English: Design note:
    - 共享缓存 key 必须包含会改变归一化、结构向量筛选或 PCA 结果的参数；
    English: - cache key , PCA resultparameter;
    - 参数直接读取 `Data_BuildPcaPriorsFull.py` 默认值，避免训练引擎另行复制一套常量；
    English: - parameterread `Data_BuildPcaPriorsFull.py` default, avoidtraining engine;
    - 最近修改时间：2026-06-17；作者：GG。
    English: - Last modified: 2026-06-17; Author: GG.
    """

    from Data_BuildPcaPriorsFull import (
        DEFAULT_BACKGROUND_FALLBACK_BASE,
        DEFAULT_BACKGROUND_FALLBACK_CHANNEL_MULTIPLIER,
        DEFAULT_BACKGROUND_MIN_CHANNEL_MULTIPLIER,
        DEFAULT_BACKGROUND_MIN_COUNT,
        DEFAULT_CHI2_QUANTILE,
        DEFAULT_COV_SHRINKAGE_ALPHA,
        DEFAULT_COV_SHRINKAGE_EPS,
        DEFAULT_CROP_HIGH1,
        DEFAULT_CROP_HIGH2,
        DEFAULT_CROP_HIGH3,
        DEFAULT_CROP_LOW,
        DEFAULT_ETA_HIGH,
        DEFAULT_ETA_LOW,
        DEFAULT_FDR_Q,
        DEFAULT_MAX_BACKGROUND_FOR_COV,
        DEFAULT_MAX_COMPONENTS,
        DEFAULT_NORM_EPS,
        DEFAULT_SELECTION_MIN_KEEP_ABS,
        DEFAULT_SELECTION_MIN_KEEP_CHANNEL_MULTIPLIER,
        DEFAULT_SELECTION_MIN_KEEP_RATIO,
        DEFAULT_SNR_DB,
    )

    return {
        "eta_high": float(DEFAULT_ETA_HIGH),
        "eta_low": float(DEFAULT_ETA_LOW),
        "max_components": None if DEFAULT_MAX_COMPONENTS is None else int(DEFAULT_MAX_COMPONENTS),
        "snr_db": float(DEFAULT_SNR_DB),
        "chi2_quantile": float(DEFAULT_CHI2_QUANTILE),
        "fdr_q": float(DEFAULT_FDR_Q),
        "selection_min_keep_ratio": float(DEFAULT_SELECTION_MIN_KEEP_RATIO),
        "selection_min_keep_channel_multiplier": int(DEFAULT_SELECTION_MIN_KEEP_CHANNEL_MULTIPLIER),
        "selection_min_keep_abs": int(DEFAULT_SELECTION_MIN_KEEP_ABS),
        "background_min_count": int(DEFAULT_BACKGROUND_MIN_COUNT),
        "background_min_channel_multiplier": int(DEFAULT_BACKGROUND_MIN_CHANNEL_MULTIPLIER),
        "background_fallback_base": int(DEFAULT_BACKGROUND_FALLBACK_BASE),
        "background_fallback_channel_multiplier": int(DEFAULT_BACKGROUND_FALLBACK_CHANNEL_MULTIPLIER),
        "max_background_for_cov": int(DEFAULT_MAX_BACKGROUND_FOR_COV),
        "cov_shrinkage_alpha": float(DEFAULT_COV_SHRINKAGE_ALPHA),
        "cov_shrinkage_eps": float(DEFAULT_COV_SHRINKAGE_EPS),
        "crop_high1": None if DEFAULT_CROP_HIGH1 is None else int(DEFAULT_CROP_HIGH1),
        "crop_high2": None if DEFAULT_CROP_HIGH2 is None else int(DEFAULT_CROP_HIGH2),
        "crop_high3": None if DEFAULT_CROP_HIGH3 is None else int(DEFAULT_CROP_HIGH3),
        "crop_low": None if DEFAULT_CROP_LOW is None else int(DEFAULT_CROP_LOW),
        "norm_eps": float(DEFAULT_NORM_EPS),
    }


def build_dataset_stable_id_trace(dataset: Any, train_indices: Sequence[int]) -> dict[str, Any]:
    """
    构建数据集样本身份追溯字段。
    English: buildsamplefield.

    设计说明:
    English: Design note:
    - 先验复用不能只看整数索引，因为换库后索引可能相同但样本不同；
    English: - , sample;
    - 这里同时记录全数据集样本身份 hash 和 Train 子集样本身份 hash；
    English: - sample hash Train sample hash;
    - 最近修改时间：2026-06-17；作者：GG。
    English: - Last modified: 2026-06-17; Author: GG.
    """

    from Train_support import get_stable_split_id_from_item

    records = list(getattr(dataset, "data_cache", dataset))
    stable_ids = [get_stable_split_id_from_item(item) for item in records]
    train_stable_ids = [stable_ids[int(index)] for index in train_indices]
    return {
        "dataset_length": int(len(stable_ids)),
        "dataset_stable_ids_sha1": compute_text_sequence_sha1(stable_ids),
        "train_stable_ids_sha1": compute_text_sequence_sha1(train_stable_ids),
    }


def write_fold_pca_prior_summary(summary: dict[str, Any], summary_path: Any) -> None:
    """
    写出 Fold PCA 先验 summary JSON。
    English: Fold PCA summary JSON.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    path = Path(str(summary_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def model_spec_requires_train_only_pca_priors(spec: ModelSpec) -> bool:
    """
    判断当前模型是否需要 Fold 内 train-only PCA 先验。
    English: determinecurrentmodel Fold train-only PCA .

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    if spec.model_family not in {"mfpchfnet", "input_ablation"}:
        return False
    return "image" in normalize_spec_active_inputs(spec)


def build_fold_pca_prior_trace(
    *,
    spec: ModelSpec,
    config: type[CommonTrainConfig],
    dataset: Any,
    split_indices: dict[str, Any],
    run_idx: int,
    run_dir: Path,
) -> dict[str, Any]:
    """
    构造 Fold 级 PCA 先验追溯字段。
    English: Fold PCA field.

    设计说明:
    English: Design note:
    - `priors_path` / `summary_path` 始终指向当前模型 Fold 目录，保证模型构建和 ONNX 导出路径不变；
    English: `summary_path` 始终指向当前模型 Fold 目录，保证模型构建和 ONNX 导出路径不变；.
    - `shared_prior_cache_key` 不包含 model_name / active_inputs，允许图像先验相同的输入消融模型跨模型复用；
    English: active_inputs，允许图像先验相同的输入消融模型跨模型复用；.
    - 共享 key 仍包含样本身份、Train 子集、图像尺寸、频层结构和构建参数，避免泄漏或结构错配；
    English: - key sample, Train , image, buildparameter, avoid;
    - 最近修改时间：2026-06-17；作者：GG。
    English: - Last modified: 2026-06-17; Author: GG.
    """

    train_indices = [int(item) for item in split_indices["train"]]
    run_dir = Path(run_dir)
    priors_path = run_dir / FOLD_TRAIN_ONLY_PCA_PRIORS_FILENAME
    summary_path = run_dir / FOLD_TRAIN_ONLY_PCA_PRIORS_SUMMARY_FILENAME
    shared_cache_root = run_dir.parent.parent / FOLD_TRAIN_ONLY_PCA_SHARED_CACHE_DIRNAME
    active_inputs = normalize_spec_active_inputs(spec)
    active_structure = resolve_required_structure_label(spec, active_inputs)
    image_size = [int(value) for value in tuple(spec.image_size)]
    builder_parameters = get_fold_pca_prior_builder_parameter_trace()
    dataset_trace = build_dataset_stable_id_trace(dataset, train_indices)
    train_indices_sha1 = compute_train_indices_sha1(train_indices)
    data_root = getattr(config, "DATA_DIR", None) or getattr(config, "DATASET_ROOT", "")
    cache_identity = {
        "cache_version": FOLD_TRAIN_ONLY_PCA_SHARED_CACHE_VERSION,
        "prior_builder": "Data_BuildPcaPriorsFull.build_pca_priors_from_training_dataset",
        "prior_build_scope": "fold_train_only",
        "leakage_guard": "validation_and_test_samples_excluded_from_prior_estimation",
        "data_root": normalize_trace_path_value(data_root),
        "gt_path": normalize_trace_path_value(getattr(config, "GT_PATH", "")),
        "tn_path": normalize_trace_path_value(getattr(config, "TN_PATH", "")),
        "target_mode": str(getattr(config, "TARGET_MODE", "soc")).lower().strip(),
        "split_seed": int(getattr(config, "SPLIT_SEED", 20260317)),
        "num_folds": int(getattr(config, "NUM_FOLDS", 8)),
        "validation_fold_offset": int(getattr(config, "VALIDATION_FOLD_OFFSET", 1)),
        "run_idx": int(run_idx),
        "test_fold": int(split_indices["test_fold"]),
        "val_fold": int(split_indices["val_fold"]),
        "train_size": int(len(train_indices)),
        "train_indices_sha1": train_indices_sha1,
        "dataset_length": int(dataset_trace["dataset_length"]),
        "dataset_stable_ids_sha1": dataset_trace["dataset_stable_ids_sha1"],
        "train_stable_ids_sha1": dataset_trace["train_stable_ids_sha1"],
        "image_size": list(image_size),
        "active_structure": active_structure,
        "builder_parameters": builder_parameters,
    }
    builder_parameters_sha1 = compute_json_sha1(builder_parameters)
    cache_identity["builder_parameters_sha1"] = builder_parameters_sha1
    shared_cache_key = compute_json_sha1(cache_identity)
    shared_cache_dir = shared_cache_root / shared_cache_key
    return {
        "enabled": True,
        "policy": "fold_train_only_pca_priors",
        "scope": "train_indices_only",
        "leakage_guard": "validation_and_test_samples_excluded_from_prior_estimation",
        "shared_prior_cache_enabled": bool(getattr(config, "SHARED_PCA_PRIOR_CACHE_ENABLED", True)),
        "shared_prior_cache_version": FOLD_TRAIN_ONLY_PCA_SHARED_CACHE_VERSION,
        "shared_prior_cache_key": shared_cache_key,
        "shared_prior_cache_root": str(shared_cache_root),
        "shared_prior_cache_dir": str(shared_cache_dir),
        "shared_priors_path": str(shared_cache_dir / FOLD_TRAIN_ONLY_PCA_PRIORS_FILENAME),
        "shared_summary_path": str(shared_cache_dir / FOLD_TRAIN_ONLY_PCA_PRIORS_SUMMARY_FILENAME),
        "shared_prior_cache_identity": cache_identity,
        "prior_builder_parameters_sha1": builder_parameters_sha1,
        "model_name": spec.name,
        "display_name": spec.display_name,
        "model_family": spec.model_family,
        "active_inputs": list(active_inputs),
        "active_structure": active_structure,
        "run_idx": int(run_idx),
        "split_seed": int(getattr(config, "SPLIT_SEED", 20260317)),
        "test_fold": int(split_indices["test_fold"]),
        "val_fold": int(split_indices["val_fold"]),
        "train_size": int(len(train_indices)),
        "val_size": int(len(split_indices["val"])),
        "test_size": int(len(split_indices["test"])),
        "train_indices_sha1": train_indices_sha1,
        "dataset_length": int(dataset_trace["dataset_length"]),
        "dataset_stable_ids_sha1": dataset_trace["dataset_stable_ids_sha1"],
        "train_stable_ids_sha1": dataset_trace["train_stable_ids_sha1"],
        "image_size": image_size,
        "priors_path": str(priors_path),
        "summary_path": str(summary_path),
    }


def fold_pca_summary_matches(summary: dict[str, Any], expected_fields: dict[str, Any]) -> bool:
    """
    判断先验 summary 是否满足给定字段约束。
    English: determine summary field.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    for key, expected_value in expected_fields.items():
        if summary.get(key) != expected_value:
            return False
    return True


def read_valid_existing_fold_pca_prior(trace: dict[str, Any]) -> dict[str, Any] | None:
    """
    若当前 Fold 已存在且 hash 匹配，则复用 train-only PCA 先验。
    English: current Fold hash , train-only PCA .

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    priors_path = Path(str(trace["priors_path"]))
    summary_path = Path(str(trace["summary_path"]))
    if not priors_path.is_file() or not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_fields = {
        "policy": trace["policy"],
        "model_name": trace["model_name"],
        "run_idx": int(trace["run_idx"]),
        "train_indices_sha1": trace["train_indices_sha1"],
        "dataset_stable_ids_sha1": trace["dataset_stable_ids_sha1"],
        "train_stable_ids_sha1": trace["train_stable_ids_sha1"],
        "active_structure": trace["active_structure"],
        "image_size": trace["image_size"],
        "shared_prior_cache_key": trace["shared_prior_cache_key"],
    }
    if not fold_pca_summary_matches(summary, expected_fields):
        return None
    summary.update(trace)
    summary["reused_existing_prior"] = True
    summary.setdefault("reuse_source", "fold_local_existing")
    summary.setdefault("shared_prior_cache_hit", False)
    return summary


def read_valid_shared_fold_pca_prior(trace: dict[str, Any]) -> dict[str, Any] | None:
    """
    从同一实验输出根目录的共享缓存读取可复用 Fold PCA 先验。
    English: Outputdirectorycacheread Fold PCA .

    设计说明:
    English: Design note:
    - 共享缓存不要求 model_name 相同，目的是允许图像先验一致的多个模型复用；
    English: - cache model_name , imagemodel;
    - 但必须匹配样本身份、Train 子集、结构、图像尺寸和构建参数；
    English: - sample, Train , , imagebuildparameter;
    - 最近修改时间：2026-06-17；作者：GG。
    English: - Last modified: 2026-06-17; Author: GG.
    """

    if not bool(trace.get("shared_prior_cache_enabled", True)):
        return None
    priors_path = Path(str(trace["shared_priors_path"]))
    summary_path = Path(str(trace["shared_summary_path"]))
    if not priors_path.is_file() or not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_fields = {
        "policy": trace["policy"],
        "shared_prior_cache_version": trace["shared_prior_cache_version"],
        "shared_prior_cache_key": trace["shared_prior_cache_key"],
        "prior_builder_parameters_sha1": trace["prior_builder_parameters_sha1"],
        "train_indices_sha1": trace["train_indices_sha1"],
        "dataset_stable_ids_sha1": trace["dataset_stable_ids_sha1"],
        "train_stable_ids_sha1": trace["train_stable_ids_sha1"],
        "active_structure": trace["active_structure"],
        "image_size": trace["image_size"],
    }
    if not fold_pca_summary_matches(summary, expected_fields):
        return None
    return summary


def copy_shared_fold_pca_prior_to_run_dir(trace: dict[str, Any], shared_summary: dict[str, Any]) -> dict[str, Any]:
    """
    将共享先验复制到当前模型 Fold 目录，并写入当前模型自己的 summary。
    English: currentmodel Fold directory, writecurrentmodel summary.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    local_priors_path = Path(str(trace["priors_path"]))
    shared_priors_path = Path(str(trace["shared_priors_path"]))
    local_priors_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shared_priors_path, local_priors_path)

    summary = dict(shared_summary)
    summary.update(trace)
    summary["reused_existing_prior"] = True
    summary["reuse_source"] = "shared_pca_prior_cache"
    summary["shared_prior_cache_hit"] = True
    summary["source_priors_path"] = str(shared_priors_path)
    summary["source_summary_path"] = str(trace["shared_summary_path"])
    summary["source_model_name"] = str(shared_summary.get("model_name", ""))
    summary["source_run_idx"] = shared_summary.get("run_idx", "")
    write_fold_pca_prior_summary(summary, trace["summary_path"])
    return summary


def update_shared_fold_pca_prior_cache(trace: dict[str, Any], local_summary: dict[str, Any]) -> None:
    """
    将当前 Fold 新构建的 PCA 先验登记到共享缓存。
    English: current Fold build PCA cache.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    if not bool(trace.get("shared_prior_cache_enabled", True)):
        return
    local_priors_path = Path(str(trace["priors_path"]))
    shared_priors_path = Path(str(trace["shared_priors_path"]))
    if not local_priors_path.is_file():
        return
    shared_priors_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_priors_path, shared_priors_path)

    shared_summary = dict(local_summary)
    shared_summary["priors_path"] = str(shared_priors_path)
    shared_summary["summary_path"] = str(trace["shared_summary_path"])
    shared_summary["reuse_source"] = "shared_cache_source_from_fold_build"
    shared_summary["shared_prior_cache_hit"] = False
    write_fold_pca_prior_summary(shared_summary, trace["shared_summary_path"])


def prepare_fold_train_only_pca_priors(
    *,
    spec: ModelSpec,
    config: type[CommonTrainConfig],
    dataset: Any,
    split_indices: dict[str, Any],
    run_idx: int,
    run_dir: Path,
) -> dict[str, Any]:
    """
    为单个 Fold 准备仅由 Train 子集估计的 PCA/归一化先验。
    English: Fold Train PCA/.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    if not model_spec_requires_train_only_pca_priors(spec):
        return {
            "enabled": False,
            "policy": "not_required_without_image_prior",
            "scope": "none",
            "model_name": spec.name,
            "model_family": spec.model_family,
            "active_inputs": list(normalize_spec_active_inputs(spec)),
        }

    trace = build_fold_pca_prior_trace(
        spec=spec,
        config=config,
        dataset=dataset,
        split_indices=split_indices,
        run_idx=run_idx,
        run_dir=run_dir,
    )
    existing = read_valid_existing_fold_pca_prior(trace)
    if existing is not None:
        print(f"    [fold-prior] 复用 train-only PCA 先验: {existing['priors_path']}")
        return existing

    shared_existing = read_valid_shared_fold_pca_prior(trace)
    if shared_existing is not None:
        copied = copy_shared_fold_pca_prior_to_run_dir(trace, shared_existing)
        print(
            f"    [fold-prior] 复用共享 train-only PCA 先验: {copied['source_priors_path']} "
            f"-> {copied['priors_path']}"
        )
        return copied

    from Data_BuildPcaPriorsFull import build_pca_priors_from_training_dataset

    print(
        f"    [fold-prior] {spec.name} Fold{run_idx:02d} 使用 Train 子集重构 PCA 先验 "
        f"(train={trace['train_size']}, val={trace['val_size']}, test={trace['test_size']})."
    )
    summary = build_pca_priors_from_training_dataset(
        dataset=dataset,
        train_indices=split_indices["train"],
        output_dir=str(run_dir),
        structure=trace["active_structure"],
        metadata=trace,
        seed=int(getattr(config, "SPLIT_SEED", 20260317)) + 3000 + int(run_idx),
    )
    summary.update(trace)
    summary["reused_existing_prior"] = False
    summary["reuse_source"] = "rebuilt_fold_train_only"
    summary["shared_prior_cache_hit"] = False
    write_fold_pca_prior_summary(summary, trace["summary_path"])
    update_shared_fold_pca_prior_cache(trace, summary)
    return summary


def is_fold_training_complete(run_dir: Path) -> bool:
    """
    判断单个 Fold 是否已经完成训练。
    English: determine Fold training.

    逻辑说明：
    English: Logic notes:
    1. 当前 V2 断点续训一直以 `metrics_summary.csv` 作为 Fold 完成标志；
    English: 1. current V2 `metrics_summary.csv` Fold ;
    2. 本函数只把原先 `run_one_fold()` 内部的跳过标准前移到模型循环层；
    English: 2. `run_one_fold()` model;
    3. 不额外要求 `run_info.json`、预测 CSV 或 checkpoint，避免改变既有续训语义。
    English: 3. `run_info.json`, CSV checkpoint, avoid.

    最近修改时间：2026-05-30；作者：ljy。
    English: Last modified: 2026-05-30; Author: ljy.
    """

    return (Path(run_dir) / "metrics_summary.csv").is_file()


def get_incomplete_run_indices(model_dir: Path, config: type[CommonTrainConfig]) -> list[int]:
    """
    返回当前模型仍需训练的 Fold 编号。
    English: returncurrentmodeltraining Fold .

    设计说明：
    English: Design note:
    - 该函数只读取磁盘完成标志，不构建 Dataset；
    English: - read, build Dataset;
    - 训练主循环据此决定是否需要加载当前模型所需的 image / hyper / nir 数据；
    English: hyper / nir 数据；.
    - 已完成模型仍可进入聚合输出和 ONNX 检查，但不会触发数据读取。
    English: - modelOutput ONNX check, read.

    最近修改时间：2026-05-30；作者：ljy。
    English: Last modified: 2026-05-30; Author: ljy.
    """

    num_runs = int(getattr(config, "NUM_RUNS", 8))
    return [
        run_idx
        for run_idx in range(1, num_runs + 1)
        if not is_fold_training_complete(Path(model_dir) / f"Fold{run_idx:02d}")
    ]


def is_model_training_complete(model_dir: Path, config: type[CommonTrainConfig]) -> bool:
    """
    判断当前模型的所有 Fold 是否已经完成。
    English: determinecurrentmodel Fold .

    最近修改时间：2026-05-30；作者：ljy。
    English: Last modified: 2026-05-30; Author: ljy.
    """

    return len(get_incomplete_run_indices(model_dir, config)) == 0


def find_next_incomplete_spec(
    model_specs: Sequence[ModelSpec],
    start_index: int,
    output_root: Path,
    config: type[CommonTrainConfig],
) -> ModelSpec | None:
    """
    查找后续第一个仍需训练的模型规格。
    English: trainingmodel.

    维护说明：
    English: :
    1. 已完成模型不需要 Dataset，因此不能因为它们的数据契约不同就提前加载或释放数据；
    English: 1. model Dataset, load;
    2. 当前模型结束后，只根据“后续第一个未完成模型”的数据契约决定保留还是释放 Dataset；
    English: 2. currentmodel, “model” Dataset;
    3. 该函数只读取各 Fold 的 `metrics_summary.csv`，不触发数据加载。
    English: 3. read Fold `metrics_summary.csv`, load.

    最近修改时间：2026-05-30；作者：ljy。
    English: Last modified: 2026-05-30; Author: ljy.
    """

    for next_spec in model_specs[start_index:]:
        next_model_dir = Path(output_root) / next_spec.name
        if not is_model_training_complete(next_model_dir, config):
            return next_spec
    return None


def resolve_fold_batch_size(spec: ModelSpec, config: type[CommonTrainConfig], model_dir: Path) -> tuple[int, dict[str, Any]]:
    """
    决定当前模型使用的 batch size。
    English: currentmodel batch size.

    断点续训优先复用旧训练进度中的 batch_size，避免菜单默认覆盖旧模型设置。
    English: training batch_size, avoidmenudefaultmodel.
    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    from Train_support import read_json_if_exists

    configured = int(spec.batch_size)
    if bool(getattr(config, "RESUME_TRAINING", False)):
        for progress_path in sorted(model_dir.glob("Fold*/training_progress.json")):
            progress = read_json_if_exists(str(progress_path))
            if progress and progress.get("batch_size"):
                previous = int(progress["batch_size"])
                return previous, {
                    "enabled": True,
                    "batch_size": previous,
                    "source": "resume_existing_model_batch",
                    "reason": f"Resume mode uses previous model batch_size={previous} from {progress_path.parent.name}/training_progress.json.",
                    "configured_batch_size": configured,
                    "previous_batch_size": previous,
                    "resume_batch_priority": "old_model_setting>menu_setting>main_entry_setting>program_default",
                    "model_spec_name": spec.name,
                    "active_inputs": list(normalize_spec_active_inputs(spec)),
                }
    return configured, {
        "enabled": True,
        "batch_size": configured,
        "source": "menu_model_spec",
        "reason": "No previous training_progress.json batch_size found; use menu ModelSpec.batch_size.",
        "configured_batch_size": configured,
        "previous_batch_size": None,
        "resume_batch_priority": "old_model_setting>menu_setting>main_entry_setting>program_default",
        "model_spec_name": spec.name,
        "active_inputs": list(normalize_spec_active_inputs(spec)),
    }


def write_auto_batch_plan(model_dir: Path, plan: dict[str, Any]) -> None:
    """
    写出当前模型 batch 决策记录。
    English: currentmodel batch .
    """

    model_dir.mkdir(parents=True, exist_ok=True)
    plan = dict(plan)
    plan["recent_modified_at"] = "2026-05-29"
    (model_dir / "auto_batch_size_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_training_device(config: type[CommonTrainConfig]) -> tuple[Any, dict[str, Any]]:
    """
    解析真实训练设备。
    English: parsetraining.

    工程意义：
    English: :
    - 正式训练优先且默认强制使用 CUDA GPU，避免 CUDA 环境异常时静默退回 CPU 长时间慢跑；
    English: - trainingdefault CUDA GPU, avoid CUDA CPU ;
    - 若确需 CPU 冒烟测试，可在 Config 中显式设置 `REQUIRE_CUDA_FOR_TRAINING = False`；
    English: - CPU , Config explicit `REQUIRE_CUDA_FOR_TRAINING = False`;
    - 返回的设备信息同时用于终端日志和 Fold 级 `run_info.json` 追溯。
    English: - return Fold `run_info.json` .

    最近修改时间：2026-05-29；作者：ljy。新增 GPU 强制检查，避免训练静默落到 CPU。
    English: Last modified: 2026-05-29; Author: ljy. GPU check, avoidtraining CPU.
    """

    import torch

    if torch.cuda.is_available():
        device_index = int(torch.cuda.current_device())
        device = torch.device(f"cuda:{device_index}")
        return device, {
            "training_device": str(device),
            "training_device_type": "cuda",
            "cuda_device_index": device_index,
            "cuda_device_name": torch.cuda.get_device_name(device_index),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
        }

    if bool(getattr(config, "REQUIRE_CUDA_FOR_TRAINING", True)):
        raise RuntimeError(
            "当前训练配置要求使用 GPU/CUDA，但 torch.cuda.is_available()=False。"
            "请检查 PyTorch 是否为 CUDA 版本、NVIDIA 驱动是否正常，或仅在明确需要 CPU 冒烟测试时设置 "
            "Config.REQUIRE_CUDA_FOR_TRAINING = False。"
        )

    device = torch.device("cpu")
    return device, {
        "training_device": str(device),
        "training_device_type": "cpu",
        "cuda_device_index": None,
        "cuda_device_name": "",
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }


def format_memory_bytes(num_bytes: int | float | None) -> str:
    """
    格式化显存字节数 / Format GPU memory bytes.
    English: Format GPU memory bytes..
    """

    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def is_cuda_out_of_memory_error(error: BaseException) -> bool:
    """
    判断异常是否为 CUDA OOM / Detect CUDA out-of-memory errors.
    English: Detect CUDA out-of-memory errors..
    """

    text = str(error).lower()
    return "cuda" in text and "out of memory" in text


def get_cuda_device_index(device: Any) -> int:
    """
    返回当前 CUDA 设备编号 / Resolve CUDA device index.
    English: Resolve CUDA device index..
    """

    import torch

    if getattr(device, "index", None) is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def estimate_adamw_state_bytes(model: Any) -> int:
    """
    估算 AdamW 首次 step 后的优化器状态显存。
    Estimate AdamW optimizer state memory after the first optimizer step.

    最近修改时间: 2026-05-29；作者: ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    说明: AdamW 通常为每个可训练参数维护 exp_avg 和 exp_avg_sq 两份状态；
    English: : AdamW trainingparameter exp_avg exp_avg_sq ;
    预检不执行 optimizer.step()，因此这里以参数 dtype 的 2 倍容量做保守补偿。
    English: optimizer.step(), parameter dtype 2 .
    """

    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += int(param.numel()) * int(param.element_size()) * 2
    return int(total)


def snapshot_batchnorm_state(model: Any) -> list[tuple[Any, dict[str, Any]]]:
    """
    记录 BatchNorm running statistics，避免显存预检污染真实训练状态。
    Snapshot BatchNorm running statistics before the memory probe.

    最近修改时间: 2026-06-07；作者: ljy。
    English: Last modified: 2026-06-07; Author: ljy.
    维护说明: BatchNorm 快照固定存放在 CPU，避免 CUDA 显存预检本身额外占用 GPU 缓冲区。
    English: : BatchNorm CPU, avoid CUDA GPU .
    """

    import torch.nn as nn

    snapshots: list[tuple[Any, dict[str, Any]]] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            state: dict[str, Any] = {}
            for attr_name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, attr_name, None)
                if value is not None:
                    state[attr_name] = value.detach().cpu().clone()
            snapshots.append((module, state))
    return snapshots


def restore_batchnorm_state(snapshots: list[tuple[Any, dict[str, Any]]]) -> None:
    """
    恢复 BatchNorm running statistics / Restore BatchNorm running statistics.
    English: Restore BatchNorm running statistics..

    最近修改时间: 2026-06-07；作者: ljy。
    English: Last modified: 2026-06-07; Author: ljy.
    维护说明: BatchNorm 快照固定存放在 CPU，避免 CUDA 显存预检本身额外占用 GPU 缓冲区。
    English: : BatchNorm CPU, avoid CUDA GPU .
    """

    import torch

    with torch.no_grad():
        for module, state in snapshots:
            for attr_name, value in state.items():
                current = getattr(module, attr_name, None)
                if current is not None:
                    current.copy_(value)


def probe_training_batch_memory(
    model: Any,
    optimizer: Any,
    dataset: Any,
    train_indices: Sequence[int],
    batch_size: int,
    loader_seed: int,
    device: Any,
    output_dim: int,
    active_inputs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    用一个训练 batch 预检当前 batch_size 的 CUDA 显存峰值。
    Probe CUDA memory peak with one training batch before real training.

    最近修改时间: 2026-06-07；作者: ljy。
    English: Last modified: 2026-06-07; Author: ljy.
    工程考虑:
    English: :
    - 预检执行 forward + backward，但不执行 optimizer.step()，避免改变模型权重；
    English: - forward + backward, optimizer.step(), avoidmodel;
    - 预检后恢复随机数状态和 BatchNorm running statistics，避免影响真实训练；
    English: - BatchNorm running statistics, avoidtraining;
    - 占用估算 = 外部/非 PyTorch 已占用显存 + PyTorch 峰值 reserved 显存 + AdamW 状态估算。
    English: - = / PyTorch + PyTorch reserved + AdamW .
    - 2026-06-07 ljy: 预检结束后先释放临时 batch / loss 张量，再恢复 BatchNorm 状态，避免清理路径再次触发 CUDA OOM。
    English: loss 张量，再恢复 BatchNorm 状态，避免清理路径再次触发 CUDA OOM.
    """

    import gc
    import torch
    import torch.nn.functional as F

    device_index = get_cuda_device_index(device)
    actual_batch_size = int(min(max(1, batch_size), max(1, len(train_indices))))
    was_training = bool(model.training)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device_index)
    batchnorm_snapshots = snapshot_batchnorm_state(model)
    optimizer.zero_grad(set_to_none=True)
    loader = None
    batch = None
    target = None
    prediction = None
    loss = None

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_index)
        free_before, total_memory = torch.cuda.mem_get_info(device_index)
        current_reserved = int(torch.cuda.memory_reserved(device_index))
        external_used = max(int(total_memory) - int(free_before) - current_reserved, 0)

        loader = make_data_loader(
            dataset,
            train_indices,
            actual_batch_size,
            False,
            loader_seed,
            device,
            active_inputs=active_inputs,
            trim_inactive_inputs=True,
        )
        batch = next(iter(loader))
        model.train()
        batch = move_batch_to_device(batch, device, active_inputs=active_inputs)
        target = batch["label"]
        if int(output_dim) == 1:
            target = target.view(-1)
        prediction = run_model_forward(model, batch)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        torch.cuda.synchronize(device_index)

        peak_reserved = int(torch.cuda.max_memory_reserved(device_index))
        peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
        optimizer_state_bytes = estimate_adamw_state_bytes(model)
        estimated_total_used = int(external_used + max(peak_reserved, peak_allocated) + optimizer_state_bytes)
        utilization = float(estimated_total_used) / float(total_memory) if int(total_memory) > 0 else 0.0
        return {
            "ok": True,
            "oom": False,
            "batch_size": int(batch_size),
            "actual_probe_batch_size": int(actual_batch_size),
            "total_memory_bytes": int(total_memory),
            "free_memory_before_probe_bytes": int(free_before),
            "external_used_bytes": int(external_used),
            "peak_reserved_bytes": int(peak_reserved),
            "peak_allocated_bytes": int(peak_allocated),
            "estimated_adamw_state_bytes": int(optimizer_state_bytes),
            "estimated_total_used_bytes": int(estimated_total_used),
            "estimated_utilization": float(utilization),
            "formatted_estimated_total_used": format_memory_bytes(estimated_total_used),
            "formatted_total_memory": format_memory_bytes(total_memory),
        }
    except Exception as error:
        if not is_cuda_out_of_memory_error(error):
            raise
        torch.cuda.empty_cache()
        return {
            "ok": False,
            "oom": True,
            "batch_size": int(batch_size),
            "actual_probe_batch_size": int(actual_batch_size),
            "error": str(error).splitlines()[0],
        }
    finally:
        optimizer.zero_grad(set_to_none=True)
        del loss, prediction, target, batch, loader
        gc.collect()
        torch.cuda.empty_cache()
        restore_batchnorm_state(batchnorm_snapshots)
        if was_training:
            model.train()
        else:
            model.eval()
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device_index)
        torch.cuda.empty_cache()


def resolve_next_memory_safe_batch_size(
    batch_size: int,
    memory_event: dict[str, Any],
    target_utilization: float,
    min_batch_size: int,
) -> int:
    """
    根据训练前显存预检事件给出下一轮 batch_size。
    Choose the next smaller batch size after a memory preflight event.

    最近修改时间: 2026-05-29；作者: ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    current = int(batch_size)
    lower_bound = int(max(1, min_batch_size))
    if current <= lower_bound:
        return lower_bound

    if bool(memory_event.get("oom")):
        memory_event["reduction_policy"] = "preflight_oom_half"
        return max(lower_bound, current // 2)

    utilization = float(memory_event.get("estimated_utilization", 1.0))
    if utilization <= 0:
        memory_event["reduction_policy"] = "invalid_utilization_half"
        return max(lower_bound, current // 2)

    overrun_ratio = max((utilization - float(target_utilization)) / max(float(target_utilization), 1e-6), 0.0)
    memory_event["target_utilization"] = float(target_utilization)
    memory_event["overrun_ratio"] = float(overrun_ratio)

    if overrun_ratio <= 0:
        memory_event["reduction_policy"] = "within_target_keep"
        return current

    if overrun_ratio <= 0.03:
        # 2026-05-29 ljy: 轻微超限只降 1，避免 85% 附近因估算波动造成过度降批。
        # EN: 2026-05-29 ljy: only 1, avoid 85% degree.
        memory_event["reduction_policy"] = "mild_overrun_minus_one"
        return max(lower_bound, current - 1)

    for limit, safety_factor, policy in (
        (0.10, 0.99, "small_overrun_proportional"),
        (0.25, 0.95, "medium_overrun_proportional"),
        (0.50, 0.90, "large_overrun_proportional"),
        (float("inf"), 0.85, "severe_overrun_aggressive"),
    ):
        if overrun_ratio <= limit:
            memory_event["reduction_policy"] = policy
            memory_event["reduction_safety_factor"] = float(safety_factor)
            proportional = int(current * float(target_utilization) / utilization * safety_factor)
            return max(lower_bound, min(current - 1, proportional))

    return lower_bound


def enforce_cuda_memory_batch_limit(
    model: Any,
    optimizer: Any,
    dataset: Any,
    train_indices: Sequence[int],
    requested_batch_size: int,
    loader_seed: int,
    device: Any,
    output_dim: int,
    config: type[CommonTrainConfig],
    active_inputs: Sequence[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    训练前检查 batch 是否会让显存占用超过阈值，必要时自动降低 batch。
    Check and reduce batch size before training if estimated VRAM use exceeds the target.

    最近修改时间: 2026-05-29；作者: ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    import torch

    if str(getattr(device, "type", "")) != "cuda" or not torch.cuda.is_available():
        return int(requested_batch_size), {
            "enabled": False,
            "reason": "CUDA is not active; GPU batch memory preflight skipped.",
        }

    target_utilization = float(getattr(config, "GPU_BATCH_MEMORY_TARGET_UTILIZATION", 0.85))
    min_batch_size = int(getattr(config, "GPU_BATCH_MEMORY_MIN_BATCH_SIZE", 1))
    candidate = int(max(min_batch_size, requested_batch_size))
    probes: list[dict[str, Any]] = []

    while True:
        probe = probe_training_batch_memory(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            train_indices=train_indices,
            batch_size=candidate,
            loader_seed=loader_seed,
            device=device,
            output_dim=output_dim,
            active_inputs=active_inputs,
        )
        probes.append(probe)
        utilization = float(probe.get("estimated_utilization", 1.0))
        if bool(probe.get("ok")) and utilization <= target_utilization:
            break
        next_candidate = resolve_next_memory_safe_batch_size(candidate, probe, target_utilization, min_batch_size)
        if next_candidate >= candidate:
            candidate = next_candidate
            break
        candidate = next_candidate

    selected_probe = probes[-1]
    selected_utilization = float(selected_probe.get("estimated_utilization", 1.0))
    if (not bool(selected_probe.get("ok"))) or selected_utilization > target_utilization:
        raise RuntimeError(
            "GPU memory preflight cannot keep estimated VRAM utilization within "
            f"{target_utilization:.0%} even at batch={int(candidate)}. "
            "Please close other GPU workloads or lower the model/input resolution."
        )
    reduced = int(candidate) < int(requested_batch_size)
    if reduced:
        print(
            "    [auto-batch] GPU memory preflight reduced batch "
            f"{int(requested_batch_size)} -> {int(candidate)} "
            f"(target<={target_utilization:.0%}, estimated={selected_utilization:.1%})."
        )
    else:
        print(
            "    [auto-batch] GPU memory preflight kept batch "
            f"{int(candidate)} (target<={target_utilization:.0%}, "
            f"estimated={selected_utilization:.1%})."
        )

    return int(candidate), {
        "enabled": True,
        "target_utilization": float(target_utilization),
        "min_batch_size": int(min_batch_size),
        "requested_batch_size": int(requested_batch_size),
        "selected_batch_size": int(candidate),
        "reduced": bool(reduced),
        "probe_count": int(len(probes)),
        "gpu_moved_inputs": list(normalize_active_inputs_for_device(active_inputs)),
        "probes": probes,
        "recent_modified_at": "2026-05-29",
        "author": "ljy",
    }


class ActiveInputSubsetDataset:
    """
    按 active_inputs 返回轻量样本视图。
    English: active_inputs returnsample.

    逻辑说明：
    English: Logic notes:
    1. SoilMultiSourceDataset 已按 active_inputs 从源头裁剪读取与缓存；
    English: 1. SoilMultiSourceDataset active_inputs readcache;
    2. 本视图继续作为训练、验证、测试和最终评价的统一 Dataset 子集包装；
    English: 2. training, , Dataset ;
    3. disk 缓存模式下本视图直接读取所需 .npy，避免 NIR/Hyper-only 阶段额外读取 image.npy；
    English: 3. disk cacheread .npy, avoid NIR/Hyper-only read image.npy;
    4. memory 缓存模式下本视图只返回所需张量，避免 DataLoader 默认 collate 拼接无关图像 batch。
    English: 4. memory cachereturn, avoid DataLoader default collate image batch.
    5. public_npz 公开库模式下 data_cache 只保存索引，本视图需调用原 Dataset 懒加载真实张量。
    English: 5. public_npz data_cache save, Dataset load.

    最近修改时间：2026-06-16；作者：GG。
    English: Last modified: 2026-06-16; Author: GG.
    """

    def __init__(self, dataset: Any, indices: Sequence[int], active_inputs: Sequence[str] | None):
        """
        初始化轻量样本视图。
        English: sample.

        输入：
        English: Input:
            dataset: 已构建完成的 SoilMultiSourceDataset。
            English: dataset: build SoilMultiSourceDataset.
            indices: 当前 Train/Validation/Test 子集索引。
            English: indices: current Train/Validation/Test .
            active_inputs: 当前模型实际启用输入源。
            English: active_inputs: currentmodelInput.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """

        self.dataset = dataset
        self.records = getattr(dataset, "data_cache", None)
        if self.records is None:
            raise TypeError("ActiveInputSubsetDataset 需要 dataset.data_cache。")
        self.indices = [int(index) for index in indices]
        self.active_inputs = normalize_active_inputs_for_device(active_inputs)

    def __len__(self) -> int:
        """
        返回当前子集样本数。
        English: returncurrentsample.
        """

        return len(self.indices)

    @staticmethod
    def _as_float_tensor(value: Any):
        """
        将 numpy / list / tensor 统一为 float32 tensor。
        English: list / tensor 统一为 float32 tensor.
        """

        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            return value.float()
        return torch.from_numpy(np.asarray(value, dtype=np.float32)).float()

    @staticmethod
    def _load_disk_tensor(path: str):
        """
        从磁盘缓存读取单个 .npy tensor。
        English: cacheread .npy tensor.
        """

        import numpy as np
        import torch

        return torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32, copy=False).copy())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        读取一个仅包含必要模态的样本。
        English: readsample.

        输出：
        English: Output:
            dict 至少包含 label，并只包含 active_inputs 声明的 hyper/nir/image。
            English: dict label, active_inputs hyper/nir/image.

        最近修改时间：2026-06-16；作者：GG。
        English: Last modified: 2026-06-16; Author: GG.
        """

        dataset_index = self.indices[int(idx)]
        item = self.records[dataset_index]
        if item.get("cache_mode") == "public_npz":
            loaded_item = self.dataset[dataset_index]
            batch_item: dict[str, Any] = {"label": self._as_float_tensor(loaded_item["label"])}
            for key in self.active_inputs:
                if key in loaded_item:
                    batch_item[key] = self._as_float_tensor(loaded_item[key])
            return batch_item

        batch_item: dict[str, Any] = {"label": self._as_float_tensor(item["label"])}
        is_disk_item = item.get("cache_mode", "memory") == "disk"
        for key in self.active_inputs:
            if is_disk_item:
                batch_item[key] = self._load_disk_tensor(item[f"{key}_path"])
            else:
                batch_item[key] = self._as_float_tensor(item[key])
        return batch_item


def make_data_loader(
    dataset: Any,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: Any,
    active_inputs: Sequence[str] | None = None,
    trim_inactive_inputs: bool = False,
):
    """
    构建 DataLoader。
    English: build DataLoader.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    维护说明：训练、验证、测试和显存预检统一按 active_inputs 裁剪 batch，
    English: : training, , active_inputs batch,.
    避免 NIR-only / Hyper-only 等输入组合在 CPU 端 collate 或 pin 无关图像张量。
    English: Hyper-only 等输入组合在 CPU 端 collate 或 pin 无关图像张量.
    """

    import torch
    from torch.utils.data import DataLoader, Subset

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader_dataset = (
        ActiveInputSubsetDataset(dataset, indices, active_inputs)
        if bool(trim_inactive_inputs) or active_inputs is not None
        else Subset(dataset, list(indices))
    )
    return DataLoader(
        loader_dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=bool(getattr(device, "type", "") == "cuda"),
    )


def normalize_active_inputs_for_device(active_inputs: Sequence[str] | None = None) -> tuple[str, ...]:
    """
    规范化需要搬到训练设备的输入源。
    English: normalizetrainingInput.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    设计说明：数据层仍保留完整 hyper/nir/image 缓存；训练层只把当前模型声明需要的有效模态搬到 GPU，
    English: Design note: hyper/nir/image cache; trainingcurrentmodel GPU,.
    避免未被模型使用的输入张量占用显存。
    English: avoidmodelInput.
    """

    if active_inputs is None:
        return ("image", "hyper", "nir")
    normalized = tuple(str(item).strip().lower() for item in active_inputs if str(item).strip())
    return normalized or ("image", "hyper", "nir")


def move_batch_to_device(batch: dict[str, Any], device: Any, active_inputs: Sequence[str] | None = None) -> dict[str, Any]:
    """
    将一个 batch 移动到训练设备。
    English: batch training.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    设计说明：label 始终进入训练设备；hyper/nir/image 仅在当前模型 active_inputs 启用时进入训练设备。
    English: Design note: label training; hyper/nir/image currentmodel active_inputs training.
    未启用模态可不出现在轻量评估 batch 中；模型 forward 会以 None 占位，且不会读取未启用分支。
    English: batch ; model forward None , read.
    """

    device_inputs = set(normalize_active_inputs_for_device(active_inputs))
    moved = {"label": batch["label"].to(device, non_blocking=True).float()}
    for key in ("hyper", "nir", "image"):
        if key not in batch:
            continue
        value = batch[key]
        if key in device_inputs:
            moved[key] = value.to(device, non_blocking=True).float()
        else:
            moved[key] = value.float()
    return moved


def run_model_forward(model: Any, batch: dict[str, Any]):
    """
    统一模型 forward 调用形式。
    English: model forward .
    """

    return model(batch.get("hyper"), batch.get("nir"), batch.get("image"))


def count_trainable_parameters(model: Any) -> dict[str, Any]:
    """
    统计模型参数量。
    English: modelparameter.
    """

    total = int(sum(param.numel() for param in model.parameters()))
    trainable = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    return {
        "params_total": total,
        "params_active_total": trainable,
        "params_registered_total": total,
        "params_m": total / 1_000_000.0,
        "params_active_m": trainable / 1_000_000.0,
        "params_registered_m": total / 1_000_000.0,
    }


def freeze_batchnorm_if_needed(model: Any, batch_size: int, config: type[CommonTrainConfig]) -> dict[str, Any]:
    """
    小 batch 时冻结 BatchNorm running statistics。
    English: batch BatchNorm running statistics.
    """

    import torch.nn as nn

    modules = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    should_freeze = bool(getattr(config, "FREEZE_BATCHNORM_WHEN_BATCH_LT_MIN_EFFECTIVE", True)) and int(batch_size) < int(
        getattr(config, "MIN_EFFECTIVE_UPDATE_BATCH_SIZE", 8)
    )
    if should_freeze:
        for module in modules:
            module.eval()
    return {
        "batchnorm_frozen_for_small_batch": bool(should_freeze),
        "batchnorm_modules_total": int(len(modules)),
        "frozen_batchnorm_modules": int(len(modules) if should_freeze else 0),
    }


def evaluate_model(
    model: Any,
    loader: Any,
    device: Any,
    output_dim: int,
    active_inputs: Sequence[str] | None = None,
) -> tuple[Any, Any, float]:
    """
    验证或测试模型并返回真值、预测和平均 MSE。
    English: modelreturnground truth, MSE.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device, active_inputs=active_inputs)
            target = batch["label"]
            if int(output_dim) == 1:
                target = target.view(-1)
            prediction = run_model_forward(model, batch)
            loss = F.mse_loss(prediction, target, reduction="sum")
            total_loss += float(loss.item())
            total_count += int(target.numel())
            all_true.append(target.detach().cpu().numpy())
            all_pred.append(prediction.detach().cpu().numpy())
    y_true = np.concatenate([item.reshape(-1, output_dim) if output_dim > 1 else item.reshape(-1, 1) for item in all_true], axis=0)
    y_pred = np.concatenate([item.reshape(-1, output_dim) if output_dim > 1 else item.reshape(-1, 1) for item in all_pred], axis=0)
    avg_loss = total_loss / max(total_count, 1)
    return y_true, y_pred, float(avg_loss)


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    output_dim: int,
    batch_size: int,
    config: type[CommonTrainConfig],
    active_inputs: Sequence[str] | None = None,
    oom_warning_state: dict[str, Any] | None = None,
) -> float:
    """
    训练一个 epoch。
    English: training epoch.
    """

    import torch.nn.functional as F

    model.train()
    freeze_batchnorm_if_needed(model, batch_size, config)
    total_loss = 0.0
    total_count = 0
    for batch_idx, batch in enumerate(loader, start=1):
        try:
            batch = move_batch_to_device(batch, device, active_inputs=active_inputs)
            target = batch["label"]
            if int(output_dim) == 1:
                target = target.view(-1)
            optimizer.zero_grad(set_to_none=True)
            prediction = run_model_forward(model, batch)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(target.numel())
            total_count += int(target.numel())
        except RuntimeError as error:
            if not is_cuda_out_of_memory_error(error):
                raise
            if oom_warning_state is not None and not bool(oom_warning_state.get("shown", False)):
                print(f"    [warning] CUDA OOM at train batch {batch_idx}; keep training without changing batch.")
                oom_warning_state["shown"] = True
    return total_loss / max(total_count, 1)


def compute_metrics_by_target(y_true: Any, y_pred: Any, target_names: Sequence[str]) -> dict[str, dict[str, float]]:
    """
    按目标变量计算指标。
    English: calculatemetric.
    """

    from Metrics_core import calculate_detailed_metrics

    metrics: dict[str, dict[str, float]] = {}
    for target_index, target_name in enumerate(target_names):
        metrics[target_name] = calculate_detailed_metrics(y_true[:, target_index], y_pred[:, target_index])
    return metrics


def select_validation_rmse(metrics: dict[str, dict[str, float]]) -> float:
    """
    返回用于 early-stop/LR patience 的验证 RMSE。
    English: return early-stop/LR patience RMSE.
    """

    values = [float(item["RMSE"]) for item in metrics.values()]
    return sum(values) / max(len(values), 1)


def select_validation_metric_mean(metrics: dict[str, dict[str, float]], metric_key: str) -> float:
    """
    返回验证集指定指标的跨目标均值。
    English: returnmetric.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    设计说明：SOC 单目标时该值就是 SOC 指标；后续扩展 SOC+TN 时，动态状态行展示两个目标的平均值。
    English: Design note: SOC SOC metric; SOC+TN , .
    """

    values = [float(item[metric_key]) for item in metrics.values() if metric_key in item]
    return sum(values) / max(len(values), 1)


def build_progress_payload(
    spec: ModelSpec,
    run_idx: int,
    epoch: int,
    train_batches_per_epoch: int,
    batch_size: int,
    best_epoch: int,
    best_val_rmse: float,
    lr_decay_cnt: int,
    lr_decay_times: int,
    current_lr: float,
    batchnorm_info: dict[str, Any],
    patience_state: dict[str, Any],
    optimizer_plan: dict[str, Any],
    optimizer_lrs: dict[str, float],
    optimizer_policy_state: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """
    构造 training_progress.json 内容。
    English: training_progress.json .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    维护说明：记录 optimizer 分组策略和当前分组学习率，保证冻结预热/分组 LR 续训时可追溯、可恢复。
    English: : optimizer currentlearning rate, ensure/ LR , .
    """

    from Train_support import get_best_model_path, get_resume_best_state_path, get_train_loss_patience_limits
    from Train_optimizer import serialize_optimizer_plan

    active_inputs = normalize_spec_active_inputs(spec)
    base_patience = int(patience_state["base_lr_patience_cycles"])
    max_multiplier = float(patience_state["train_loss_patience_max_multiplier"])
    return {
        "model_name": spec.display_name,
        "run_idx": int(run_idx),
        "laplacian_levels": 3,
        "active_inputs": format_active_inputs_text(active_inputs),
        "input_setting": format_input_setting(active_inputs),
        "last_validated_epoch": int(epoch),
        "last_validated_epoch_progress": float(epoch),
        "last_validated_batch_idx": int(train_batches_per_epoch),
        "train_batches_per_epoch": int(train_batches_per_epoch),
        "validation_batch_interval": int(train_batches_per_epoch),
        "effective_val_interval_epochs": 1.0,
        **batchnorm_info,
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "lr_decay_cnt": int(lr_decay_cnt),
        "lr_decay_times": int(lr_decay_times),
        "batch_size": int(batch_size),
        "base_lr_patience_cycles": base_patience,
        "lr_patience_cycles": int(patience_state["current_lr_patience_cycles"]),
        "lr_patience_multiplier": 1.0,
        "lr_patience_small_batch_threshold": int(patience_state["lr_patience_small_batch_threshold"]),
        "lr_patience_reason": "grad_accum_steps_eq_1",
        "best_train_epoch_loss": patience_state.get("best_train_epoch_loss"),
        "train_loss_patience_bonus_cycles": int(patience_state["train_loss_patience_bonus_cycles"]),
        "current_lr_patience_cycles": int(patience_state["current_lr_patience_cycles"]),
        "train_loss_patience_max_cycles": int(get_train_loss_patience_limits(base_patience, max_multiplier)["max_patience_cycles"]),
        "train_loss_patience_reset_reason": patience_state.get("train_loss_patience_reset_reason", "none"),
        "current_lr": float(current_lr),
        "optimizer_plan": serialize_optimizer_plan(optimizer_plan),
        "optimizer_lrs": {str(key): float(value) for key, value in optimizer_lrs.items()},
        **optimizer_policy_state,
        "resume_policy": "from_best_validation_state",
        "best_model_path": get_best_model_path(str(run_dir)),
        "resume_state_path": get_resume_best_state_path(str(run_dir)),
    }


def save_training_checkpoint(
    model: Any,
    optimizer: Any,
    run_dir: Path,
    epoch: int,
    best_val_rmse: float,
    lr_decay_cnt: int,
    lr_decay_times: int,
    extra_state: dict[str, Any],
) -> None:
    """
    保存 best_model.pth 和 resume_from_best_state.pth。
    English: save best_model.pth resume_from_best_state.pth.
    """

    import torch
    from Train_support import get_best_model_path, get_resume_best_state_path

    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = get_best_model_path(str(run_dir))
    resume_state_path = get_resume_best_state_path(str(run_dir))
    torch.save(model.state_dict(), best_model_path)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rmse": float(best_val_rmse),
            "lr_decay_cnt": int(lr_decay_cnt),
            "lr_decay_times": int(lr_decay_times),
            "extra_state": dict(extra_state),
        },
        resume_state_path,
    )


def load_resume_checkpoint_if_available(model: Any, optimizer: Any, run_dir: Path, config: type[CommonTrainConfig]) -> dict[str, Any]:
    """
    按当前 run_dir 读取断点。
    English: current run_dir read.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    维护说明：optimizer 策略支持分组学习率后，旧单组 optimizer checkpoint 可能无法直接载入新分组结构；
    English: : optimizer policylearning rate, optimizer checkpoint ;
    此时保留模型权重续训，并按 progress/current_lr 恢复学习率比例。
    English: model, progress/current_lr learning rate.
    """

    import torch
    from Train_support import get_resume_best_state_path, read_json_if_exists
    from Train_optimizer import get_primary_optimizer_lr, restore_optimizer_lrs_from_progress

    state = {
        "loaded": False,
        "start_epoch": 1,
        "best_epoch": 0,
        "best_val_rmse": float("inf"),
        "lr_decay_cnt": 0,
        "lr_decay_times": 0,
        "current_lr": float(getattr(config, "LEARNING_RATE", 1e-4)),
        "patience_extra": {},
    }
    if not bool(getattr(config, "RESUME_TRAINING", False)):
        return state

    resume_state_path = Path(get_resume_best_state_path(str(run_dir)))
    progress = read_json_if_exists(str(run_dir / "training_progress.json"))
    if not resume_state_path.is_file():
        return state

    checkpoint = torch.load(resume_state_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if checkpoint.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (RuntimeError, ValueError) as error:
            print(f"    [warning] optimizer state 与当前分组策略不匹配，仅恢复模型权重: {error}")
    restore_optimizer_lrs_from_progress(optimizer, progress or checkpoint.get("extra_state", {}))

    state.update({
        "loaded": True,
        "start_epoch": int((progress or {}).get("last_validated_epoch", checkpoint.get("epoch", 0))) + 1,
        "best_epoch": int((progress or {}).get("best_epoch", checkpoint.get("epoch", 0))),
        "best_val_rmse": float((progress or {}).get("best_val_rmse", checkpoint.get("best_val_rmse", float("inf")))),
        "lr_decay_cnt": int((progress or {}).get("lr_decay_cnt", checkpoint.get("lr_decay_cnt", 0))),
        "lr_decay_times": int((progress or {}).get("lr_decay_times", checkpoint.get("lr_decay_times", 0))),
        "current_lr": float(get_primary_optimizer_lr(optimizer)),
        "patience_extra": dict(progress or {}),
    })
    return state


def resolve_lr_patience_stop_reason(lr_decay_times: int, max_lr_decays: int) -> Optional[str]:
    """
    判断 LR patience 触发时是否应直接结束当前 Fold。
    English: determine LR patience current Fold.

    逻辑说明：
    English: Logic notes:
    1. `lr_decay_times == 0` 表示当前还在基础 LR 阶段；此时若未达到用户允许的最大降 LR 次数，仍允许执行第一次降 LR；
    English: 1. `lr_decay_times == 0` current LR ; LR , LR;
    2. `lr_decay_times >= 1` 表示至少已经完成过一次降 LR 后的训练观察；如果此时验证 RMSE 仍没有刷新 best，
    English: 2. `lr_decay_times >= 1` LR training; RMSE best,.
       说明第一轮低 LR 训练未带来优化，直接结束当前 Fold，不再继续第二轮/第三轮降 LR；
       English: LR training, current Fold, / LR;
    3. `MAX_LR_DECAYS <= 0` 的历史含义保持不变：基础 LR 阶段 patience 触发后立即停止。
    English: 3. `MAX_LR_DECAYS <= 0` : LR patience .

    最近修改时间：2026-05-30；作者：ljy。
    English: Last modified: 2026-05-30; Author: ljy.
    """

    lr_decay_times = int(lr_decay_times)
    max_lr_decays = int(max_lr_decays)
    if max_lr_decays <= 0 and lr_decay_times >= max_lr_decays:
        return "max_lr_decays_reached"
    if lr_decay_times >= 1:
        return "first_lr_decay_no_validation_improvement"
    if lr_decay_times >= max_lr_decays:
        return "max_lr_decays_reached"
    return None


def write_fold_outputs(
    spec: ModelSpec,
    config: type[CommonTrainConfig],
    run_dir: Path,
    run_idx: int,
    split_indices: dict[str, Any],
    model: Any,
    loaders_for_eval: dict[str, Any],
    dataset: Any,
    device: Any,
    best_epoch: int,
    best_val_rmse: float,
    batch_size: int,
    batchnorm_info: dict[str, Any],
    patience_state: dict[str, Any],
    optimizer_plan: dict[str, Any],
    optimizer_lrs: dict[str, float],
    optimizer_policy_state: dict[str, Any],
    fold_pca_prior: dict[str, Any] | None = None,
) -> None:
    """
    保存 Fold 级指标、测试预测和 run_info。
    English: save Fold metric, run_info.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    维护说明：
    English: :
    - 输出中同步记录 optimizer policy、分组学习率和冻结状态，便于 Compare/SwinV2 微调策略复核；
    English: - Output optimizer policy, learning rate, Compare/SwinV2 ;
    - 新增 Fold 级 train-only PCA 先验追溯字段，用于后续 ONNX 代表 Fold 重建。
    English: - Fold train-only PCA field, ONNX Fold .
    """

    import pandas as pd
    import torch
    from Metrics_core import save_csv_rounded
    from Train_support import get_best_model_path, get_resume_best_state_path
    from Train_optimizer import serialize_optimizer_plan

    best_model_path = Path(get_best_model_path(str(run_dir)))
    if best_model_path.is_file():
        model.load_state_dict(torch.load(best_model_path, map_location=device), strict=True)

    target_names = get_target_names(config)
    output_dim = len(target_names)
    active_inputs = resolve_model_active_inputs(model, spec)
    optimizer_plan_summary = serialize_optimizer_plan(optimizer_plan)
    fold_pca_prior = dict(fold_pca_prior or {"enabled": False, "policy": "not_recorded"})
    metrics_by_subset: dict[str, dict[str, dict[str, float]]] = {}
    predictions_by_subset = {}
    for subset_name, loader in loaders_for_eval.items():
        y_true, y_pred, _ = evaluate_model(model, loader, device, output_dim, active_inputs=active_inputs)
        metrics_by_subset[subset_name] = compute_metrics_by_target(y_true, y_pred, target_names)
        predictions_by_subset[subset_name] = (y_true, y_pred)

    params = count_trainable_parameters(model)
    rows = []
    for subset_name, target_metrics in metrics_by_subset.items():
        for target_name, metrics in target_metrics.items():
            rows.append({
                "Model": spec.display_name,
                "Fold": int(run_idx),
                "Subset": subset_name,
                "Target": target_name,
                "Input_Size": format_size_tag(spec.image_size),
                "Ablation_Mode": spec.model_family,
                "Pyramid_Setting": format_pyramid_setting_for_spec(spec, active_inputs),
                "Input_Setting": format_input_setting(active_inputs),
                "Active_Inputs": format_active_inputs_text(active_inputs),
                "Result_Source": "trained_input_ablation" if spec.model_family == "input_ablation" else "trained",
                "Batch_Size": int(batch_size),
                "Base_LR_Patience_Cycles": int(patience_state["base_lr_patience_cycles"]),
                "LR_Patience_Cycles": int(patience_state["current_lr_patience_cycles"]),
                "LR_Patience_Multiplier": 1.0,
                "LR_Patience_Small_Batch_Threshold": int(patience_state["lr_patience_small_batch_threshold"]),
                "LR_Patience_Reason": "grad_accum_steps_eq_1",
                "Optimizer_Policy": optimizer_plan_summary["policy"],
                "Optimizer_Head_LR": optimizer_plan_summary.get("head_lr"),
                "Optimizer_Backbone_LR": optimizer_plan_summary.get("backbone_lr"),
                "Optimizer_Freeze_Backbone_Epochs": int(optimizer_plan_summary["freeze_backbone_epochs"]),
                "Optimizer_Backbone_Frozen": bool(optimizer_policy_state.get("backbone_frozen_by_optimizer_policy", False)),
                "BatchNorm_Frozen_For_Small_Batch": bool(batchnorm_info["batchnorm_frozen_for_small_batch"]),
                "BatchNorm_Modules_Total": int(batchnorm_info["batchnorm_modules_total"]),
                "Frozen_BatchNorm_Modules": int(batchnorm_info["frozen_batchnorm_modules"]),
                "Split_Seed": int(getattr(config, "SPLIT_SEED", 20260317)),
                "Test_Fold": int(split_indices["test_fold"]),
                "Val_Fold": int(split_indices["val_fold"]),
                "Train_Loader_Seed": int(getattr(config, "SPLIT_SEED", 20260317)) + 1000 + int(run_idx),
                "Weight_Seed": int(getattr(config, "SPLIT_SEED", 20260317)) + 2000 + int(run_idx),
                "Fold_PCA_Prior_Policy": str(fold_pca_prior.get("policy", "")),
                "Fold_PCA_Prior_Path": str(fold_pca_prior.get("priors_path", "")),
                "Fold_PCA_Prior_Train_Size": fold_pca_prior.get("train_size", ""),
                "Fold_PCA_Prior_Train_Indices_SHA1": str(fold_pca_prior.get("train_indices_sha1", "")),
                "Best_Val_RMSE": float(best_val_rmse),
                **params,
                "FLOPs_G": 0.0,
                **metrics,
            })

    save_csv_rounded(pd.DataFrame(rows), str(run_dir / "metrics_summary.csv"), int(getattr(config, "EXPORT_DECIMALS", 6)))

    records = getattr(dataset, "data_cache", [])
    test_true, test_pred = predictions_by_subset["Test"]
    prediction_rows = []
    for row_index, dataset_index in enumerate(split_indices["test"]):
        item = records[int(dataset_index)]
        row = {
            "Sample_Index": int(dataset_index),
            "Sample_Name": item.get("sample_name", ""),
            "Core_ID": item.get("core_id", ""),
        }
        for target_index, target_name in enumerate(target_names):
            row[f"True_{target_name}"] = float(test_true[row_index, target_index])
            row[f"Predicted_{target_name}"] = float(test_pred[row_index, target_index])
            row[f"Absolute_Error_{target_name}"] = abs(row[f"True_{target_name}"] - row[f"Predicted_{target_name}"])
        prediction_rows.append(row)
    save_csv_rounded(
        pd.DataFrame(prediction_rows),
        str(run_dir / f"test_predictions_fold_{int(run_idx):02d}_{str(getattr(config, 'TARGET_MODE', 'soc')).lower()}.csv"),
        int(getattr(config, "EXPORT_DECIMALS", 6)),
    )

    run_info = {
        "model_name": spec.display_name,
        "engineering_model_name": spec.name,
        "engineering_model_version": "Model_MFPCHFNet",
        "paper_model_name_note": "Paper name remains MFPC-HFNet without V2.",
        "run_idx": int(run_idx),
        "ablation_mode": spec.model_family,
        "laplacian_levels": 3,
        "pyramid_label": format_pyramid_setting_for_spec(spec, active_inputs),
        "input_setting": format_input_setting(active_inputs),
        "active_inputs": format_active_inputs_text(active_inputs),
        "result_source": "trained_input_ablation" if spec.model_family == "input_ablation" else "trained",
        "image_hw": list(spec.image_size),
        "batch_size": int(batch_size),
        "training_device": str(device),
        "training_device_type": str(getattr(device, "type", device)),
        "cuda_device_name": torch.cuda.get_device_name(device.index if device.index is not None else torch.cuda.current_device())
        if str(getattr(device, "type", "")) == "cuda" and torch.cuda.is_available()
        else "",
        "input_size_tag": format_size_tag(spec.image_size),
        "split_seed": int(getattr(config, "SPLIT_SEED", 20260317)),
        "loader_seed": int(getattr(config, "SPLIT_SEED", 20260317)) + 1000 + int(run_idx),
        "weight_seed": int(getattr(config, "SPLIT_SEED", 20260317)) + 2000 + int(run_idx),
        "train_size": len(split_indices["train"]),
        "val_size": len(split_indices["val"]),
        "test_size": len(split_indices["test"]),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        **params,
        "pca_prior_policy": str(fold_pca_prior.get("policy", "")),
        "pca_priors_path": str(fold_pca_prior.get("priors_path", "")),
        "pca_priors_summary_path": str(fold_pca_prior.get("summary_path", "")),
        "pca_prior_train_size": fold_pca_prior.get("train_size", ""),
        "pca_prior_train_indices_sha1": str(fold_pca_prior.get("train_indices_sha1", "")),
        "fold_pca_prior": fold_pca_prior,
        "optimizer_plan": optimizer_plan_summary,
        "optimizer_lrs": {str(key): float(value) for key, value in optimizer_lrs.items()},
        **optimizer_policy_state,
        "flops_g": 0.0,
        **batchnorm_info,
        "resume_policy": "from_best_validation_state",
        "checkpoint_cleanup_enabled": bool(getattr(config, "CLEANUP_COMPLETED_FOLD_CHECKPOINTS", True)),
        "keep_completed_fold_best_model": bool(getattr(config, "KEEP_COMPLETED_FOLD_BEST_MODEL", True)),
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(getattr(config, "CLEANUP_COMPLETED_FOLD_CHECKPOINTS", True)):
        resume_state_path = Path(get_resume_best_state_path(str(run_dir)))
        if resume_state_path.is_file():
            resume_state_path.unlink()


def run_one_fold(
    spec: ModelSpec,
    config: type[CommonTrainConfig],
    dataset: Any,
    split_indices: dict[str, Any],
    run_idx: int,
    run_dir: Path,
    batch_size: int,
    device: Any,
) -> int:
    """
    训练或续训单个 Fold。
    English: training Fold.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    维护说明：LR patience 触发时，基础 LR 阶段仍允许第一次降 LR；第一次降 LR 后若验证 best 未刷新，
    English: : LR patience , LR LR; LR best ,.
    则直接结束当前 Fold，不再继续后续降 LR 轮次，避免无收益训练继续消耗时间。
    English: current Fold, LR , avoidtraining.
    2026-06-17 GG：模型构建前先用当前 Fold 的 Train 子集重构 PCA/归一化先验，Validation/Test 不参与估计。
    English: 2026-06-17 GG: modelbuildcurrent Fold Train PCA/, Validation/Test .
    """

    from Train_support import (
        append_validation_history,
        get_best_model_path,
        release_cuda_memory,
        reset_train_loss_patience_bonus,
        save_training_progress_json,
        set_global_seed,
        update_train_loss_patience_after_epoch,
    )
    from Train_optimizer import (
        build_optimizer_for_spec,
        decay_optimizer_lrs,
        get_optimizer_lrs,
        get_optimizer_memory_preflight_epoch,
        get_primary_optimizer_lr,
        serialize_optimizer_plan,
        sync_optimizer_policy_for_epoch,
    )

    metrics_path = run_dir / "metrics_summary.csv"
    if metrics_path.is_file():
        print(f"    [skip] {spec.name} Fold{run_idx:02d} 已存在 metrics_summary.csv。")
        return int(batch_size)

    run_dir.mkdir(parents=True, exist_ok=True)
    split_seed = int(getattr(config, "SPLIT_SEED", 20260317))
    loader_seed = split_seed + 1000 + int(run_idx)
    weight_seed = split_seed + 2000 + int(run_idx)
    set_global_seed(weight_seed)

    fold_pca_prior = prepare_fold_train_only_pca_priors(
        spec=spec,
        config=config,
        dataset=dataset,
        split_indices=split_indices,
        run_idx=run_idx,
        run_dir=run_dir,
    )
    if bool(fold_pca_prior.get("enabled")):
        spec = clone_model_spec(spec, priors_path=str(fold_pca_prior["priors_path"]))

    model = build_model_from_spec(spec, config).to(device)
    model_active_inputs = resolve_model_active_inputs(model, spec)
    output_dim = get_output_dim(config)
    optimizer, optimizer_plan = build_optimizer_for_spec(model, spec, config)
    optimizer_plan_summary = serialize_optimizer_plan(optimizer_plan)

    resume_state = load_resume_checkpoint_if_available(model, optimizer, run_dir, config)

    batch_plan_path = run_dir.parent / "auto_batch_size_plan.json"
    batch_plan = {}
    if batch_plan_path.is_file():
        try:
            batch_plan = json.loads(batch_plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            batch_plan = {}
    preflight_epoch = get_optimizer_memory_preflight_epoch(optimizer_plan, int(resume_state["start_epoch"]))
    sync_optimizer_policy_for_epoch(model, optimizer_plan, preflight_epoch)
    batch_size, memory_plan = enforce_cuda_memory_batch_limit(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        train_indices=split_indices["train"],
        requested_batch_size=batch_size,
        loader_seed=loader_seed,
        device=device,
        output_dim=output_dim,
        config=config,
        active_inputs=model_active_inputs,
    )
    optimizer_policy_state = sync_optimizer_policy_for_epoch(model, optimizer_plan, int(resume_state["start_epoch"]))
    batch_plan = dict(batch_plan)
    batch_plan["batch_size"] = int(batch_size)
    batch_plan["effective_batch_size"] = int(batch_size)
    batch_plan["cuda_memory_preflight"] = memory_plan
    if bool(memory_plan.get("reduced")):
        batch_plan["source"] = f"{batch_plan.get('source', 'unknown')}+cuda_memory_preflight"
        batch_plan["reason"] = (
            f"{batch_plan.get('reason', '')} GPU memory preflight reduced batch to keep estimated VRAM utilization "
            f"within {float(memory_plan.get('target_utilization', 0.85)):.0%}."
        ).strip()
    write_auto_batch_plan(run_dir.parent, batch_plan)

    train_loader = make_data_loader(
        dataset,
        split_indices["train"],
        batch_size,
        True,
        loader_seed,
        device,
        active_inputs=model_active_inputs,
        trim_inactive_inputs=True,
    )
    val_loader = make_data_loader(
        dataset,
        split_indices["val"],
        batch_size,
        False,
        loader_seed,
        device,
        active_inputs=model_active_inputs,
        trim_inactive_inputs=True,
    )
    test_loader = make_data_loader(
        dataset,
        split_indices["test"],
        batch_size,
        False,
        loader_seed,
        device,
        active_inputs=model_active_inputs,
        trim_inactive_inputs=True,
    )
    eval_loaders = {
        "Train": make_data_loader(
            dataset,
            split_indices["train"],
            batch_size,
            False,
            loader_seed,
            device,
            active_inputs=model_active_inputs,
            trim_inactive_inputs=True,
        ),
        "Validation": val_loader,
        "Test": test_loader,
    }
    train_batches_per_epoch = len(train_loader)
    batchnorm_info = freeze_batchnorm_if_needed(model, batch_size, config)

    base_patience = int(getattr(config, "LR_PATIENCE_CYCLES", 25))
    progress_extra = dict(resume_state.get("patience_extra") or {})
    patience_state = {
        "base_lr_patience_cycles": base_patience,
        "lr_patience_small_batch_threshold": int(getattr(config, "LR_PATIENCE_SMALL_BATCH_THRESHOLD", 8)),
        "best_train_epoch_loss": float(progress_extra.get("best_train_epoch_loss", float("inf"))),
        "train_loss_patience_bonus_cycles": int(progress_extra.get("train_loss_patience_bonus_cycles", 0)),
        "current_lr_patience_cycles": int(progress_extra.get("current_lr_patience_cycles", base_patience)),
        "train_loss_patience_max_multiplier": float(getattr(config, "TRAIN_LOSS_PATIENCE_MAX_MULTIPLIER", 4.0)),
        "train_loss_patience_reset_reason": progress_extra.get("train_loss_patience_reset_reason", "none"),
    }

    start_epoch = int(resume_state["start_epoch"])
    best_epoch = int(resume_state["best_epoch"])
    best_val_rmse = float(resume_state["best_val_rmse"])
    lr_decay_cnt = int(resume_state["lr_decay_cnt"])
    lr_decay_times = int(resume_state["lr_decay_times"])
    max_epochs = int(getattr(config, "MAX_EPOCHS", 1000))
    max_lr_decays = int(getattr(config, "MAX_LR_DECAYS", 3))

    print(
        f"    [fold] {spec.name} Fold{run_idx:02d} start_epoch={start_epoch}, "
        f"best_val_rmse={best_val_rmse:.6f}, batch={batch_size}, device={device}"
    )
    if optimizer_plan_summary["policy"] != "default_adamw":
        print(
            f"    [optimizer] policy={optimizer_plan_summary['policy']} | "
            f"head_lr={optimizer_plan_summary.get('head_lr')} | "
            f"backbone_lr={optimizer_plan_summary.get('backbone_lr')} | "
            f"freeze_backbone_epochs={optimizer_plan_summary['freeze_backbone_epochs']}"
        )

    should_finish = False
    should_finish_reason = ""
    train_live_status_width = 0
    oom_warning_state: dict[str, Any] = {"shown": False}
    for epoch in range(start_epoch, max_epochs + 1):
        optimizer_policy_state = sync_optimizer_policy_for_epoch(model, optimizer_plan, epoch)
        avg_train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            output_dim=output_dim,
            batch_size=batch_size,
            config=config,
            active_inputs=model_active_inputs,
            oom_warning_state=oom_warning_state,
        )
        if epoch % int(getattr(config, "VAL_INTERVAL", 1)) != 0:
            train_live_status_width = write_terminal_live_status(
                (
                    f"    [train] Fold{run_idx:02d} E{epoch}/{max_epochs} | "
                    f"loss={float(avg_train_loss):.6f} | val=wait | "
                    f"best={float(best_val_rmse):.6f} | lr={float(get_primary_optimizer_lr(optimizer)):.2e}"
                ),
                previous_width=train_live_status_width,
            )
            continue

        y_val_true, y_val_pred, _ = evaluate_model(
            model,
            val_loader,
            device,
            output_dim,
            active_inputs=model_active_inputs,
        )
        val_metrics = compute_metrics_by_target(y_val_true, y_val_pred, get_target_names(config))
        val_rmse = select_validation_rmse(val_metrics)
        val_r2 = select_validation_metric_mean(val_metrics, "R2_Score")
        val_ev = select_validation_metric_mean(val_metrics, "Explained_Variance")
        is_best = val_rmse < best_val_rmse

        if is_best:
            best_epoch = int(epoch)
            best_val_rmse = float(val_rmse)
            lr_decay_cnt = 0
            patience_state.update(reset_train_loss_patience_bonus(base_patience))
            patience_state["best_train_epoch_loss"] = min(float(avg_train_loss), float(patience_state["best_train_epoch_loss"]))
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                run_dir=run_dir,
                epoch=epoch,
                best_val_rmse=best_val_rmse,
                lr_decay_cnt=lr_decay_cnt,
                lr_decay_times=lr_decay_times,
                extra_state={
                    "model_name": spec.display_name,
                    "active_inputs": format_active_inputs_text(normalize_spec_active_inputs(spec)),
                    "input_setting": format_input_setting(normalize_spec_active_inputs(spec)),
                    "run_idx": int(run_idx),
                    "batch_size": int(batch_size),
                    "optimizer_plan": optimizer_plan_summary,
                    "optimizer_lrs": get_optimizer_lrs(optimizer),
                    "fold_pca_prior": dict(fold_pca_prior),
                    **optimizer_policy_state,
                    "best_epoch_progress": float(epoch),
                    **batchnorm_info,
                    **patience_state,
                },
            )
        else:
            lr_decay_cnt += 1
            patience_update = update_train_loss_patience_after_epoch(
                epoch_train_loss=float(avg_train_loss),
                best_train_epoch_loss=float(patience_state["best_train_epoch_loss"]),
                current_bonus_cycles=int(patience_state["train_loss_patience_bonus_cycles"]),
                base_patience_cycles=base_patience,
                bonus_cycles=int(getattr(config, "TRAIN_LOSS_PATIENCE_BONUS_CYCLES", 25)),
                max_multiplier=float(getattr(config, "TRAIN_LOSS_PATIENCE_MAX_MULTIPLIER", 4.0)),
                enabled=bool(getattr(config, "TRAIN_LOSS_PATIENCE_BONUS_ENABLED", True)),
            )
            patience_state.update(patience_update)
            patience_state["train_loss_patience_max_multiplier"] = float(getattr(config, "TRAIN_LOSS_PATIENCE_MAX_MULTIPLIER", 4.0))
            patience_state["lr_patience_small_batch_threshold"] = int(getattr(config, "LR_PATIENCE_SMALL_BATCH_THRESHOLD", 8))

        progress_payload = build_progress_payload(
            spec=spec,
            run_idx=run_idx,
            epoch=epoch,
            train_batches_per_epoch=train_batches_per_epoch,
            batch_size=batch_size,
            best_epoch=best_epoch,
            best_val_rmse=best_val_rmse,
            lr_decay_cnt=lr_decay_cnt,
            lr_decay_times=lr_decay_times,
            current_lr=get_primary_optimizer_lr(optimizer),
            batchnorm_info=batchnorm_info,
            patience_state=patience_state,
            optimizer_plan=optimizer_plan,
            optimizer_lrs=get_optimizer_lrs(optimizer),
            optimizer_policy_state=optimizer_policy_state,
            run_dir=run_dir,
        )
        save_training_progress_json(str(run_dir), progress_payload)
        history_row = {
            "Epoch": int(epoch),
            "Train_Loss": float(avg_train_loss),
            "LR": float(get_primary_optimizer_lr(optimizer)),
            "Optimizer_LRs": json.dumps(get_optimizer_lrs(optimizer), ensure_ascii=False),
            "Optimizer_Policy": optimizer_plan_summary["policy"],
            "Optimizer_Head_LR": optimizer_plan_summary.get("head_lr"),
            "Optimizer_Backbone_LR": optimizer_plan_summary.get("backbone_lr"),
            "Optimizer_Freeze_Backbone_Epochs": int(optimizer_plan_summary["freeze_backbone_epochs"]),
            "Optimizer_Backbone_Frozen": bool(optimizer_policy_state.get("backbone_frozen_by_optimizer_policy", False)),
            "Val_Mean_RMSE": float(val_rmse),
            "Is_Best": int(bool(is_best)),
            "LR_Decay_Cnt": int(lr_decay_cnt),
            "LR_Decay_Times": int(lr_decay_times),
            "Epoch_Progress": float(epoch),
            "Train_Batch_Idx": int(train_batches_per_epoch),
            "Train_Batches_Per_Epoch": int(train_batches_per_epoch),
            "Validation_Batch_Interval": int(train_batches_per_epoch),
            "Effective_Val_Interval_Epochs": 1.0,
            "BatchNorm_Frozen_For_Small_Batch": bool(batchnorm_info["batchnorm_frozen_for_small_batch"]),
            "BatchNorm_Modules_Total": int(batchnorm_info["batchnorm_modules_total"]),
            "Frozen_BatchNorm_Modules": int(batchnorm_info["frozen_batchnorm_modules"]),
            "Base_LR_Patience_Cycles": int(base_patience),
            "LR_Patience_Cycles": int(patience_state["current_lr_patience_cycles"]),
            "LR_Patience_Multiplier": 1.0,
            "LR_Patience_Small_Batch_Threshold": int(patience_state["lr_patience_small_batch_threshold"]),
            "LR_Patience_Reason": "grad_accum_steps_eq_1",
            "Best_Train_Epoch_Loss": patience_state["best_train_epoch_loss"],
            "Train_Loss_Patience_Bonus_Cycles": int(patience_state["train_loss_patience_bonus_cycles"]),
            "Current_LR_Patience_Cycles": int(patience_state["current_lr_patience_cycles"]),
            "Train_Loss_Patience_Max_Cycles": int(
                round(base_patience * float(getattr(config, "TRAIN_LOSS_PATIENCE_MAX_MULTIPLIER", 4.0)))
            ),
            "Train_Loss_Patience_Reset_Reason": patience_state.get("train_loss_patience_reset_reason", "none"),
        }
        for target_name, metrics in val_metrics.items():
            history_row[f"Val_{target_name}_RMSE"] = float(metrics["RMSE"])
            history_row[f"Val_{target_name}_R2"] = float(metrics["R2_Score"])
            history_row[f"Val_{target_name}_EV"] = float(metrics["Explained_Variance"])
        append_validation_history(str(run_dir), history_row)

        best_mark = "best" if is_best else f"wait={lr_decay_cnt}/{int(patience_state['current_lr_patience_cycles'])}"
        train_live_status_width = write_terminal_live_status(
            (
                f"    [train] Fold{run_idx:02d} E{epoch}/{max_epochs} | "
                f"loss={float(avg_train_loss):.6f} | val={float(val_rmse):.6f} | "
                f"r2={float(val_r2):.4f} | ev={float(val_ev):.4f} | "
                f"best={float(best_val_rmse):.6f} | lr={float(get_primary_optimizer_lr(optimizer)):.2e} | {best_mark}"
            ),
            previous_width=train_live_status_width,
        )
        if is_best:
            finish_terminal_live_status(train_live_status_width)
            train_live_status_width = 0

        if lr_decay_cnt >= int(patience_state["current_lr_patience_cycles"]):
            stop_reason = resolve_lr_patience_stop_reason(lr_decay_times, max_lr_decays)
            if stop_reason:
                should_finish = True
                should_finish_reason = stop_reason
                break
            decay_optimizer_lrs(optimizer, float(getattr(config, "LR_DECAY_FACTOR", 0.5)))
            lr_decay_times += 1
            lr_decay_cnt = 0
            patience_state.update(reset_train_loss_patience_bonus(base_patience))

    finish_terminal_live_status(train_live_status_width)

    if not Path(get_best_model_path(str(run_dir))).is_file():
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            run_dir=run_dir,
            epoch=max(start_epoch, 1),
            best_val_rmse=best_val_rmse if best_val_rmse < float("inf") else 0.0,
            lr_decay_cnt=lr_decay_cnt,
            lr_decay_times=lr_decay_times,
            extra_state={
                "optimizer_plan": optimizer_plan_summary,
                "optimizer_lrs": get_optimizer_lrs(optimizer),
                "fold_pca_prior": dict(fold_pca_prior),
                **optimizer_policy_state,
            },
        )

    write_fold_outputs(
        spec=spec,
        config=config,
        run_dir=run_dir,
        run_idx=run_idx,
        split_indices=split_indices,
        model=model,
        loaders_for_eval=eval_loaders,
        dataset=dataset,
        device=device,
        best_epoch=best_epoch,
        best_val_rmse=best_val_rmse,
        batch_size=batch_size,
        batchnorm_info=batchnorm_info,
        patience_state=patience_state,
        optimizer_plan=optimizer_plan,
        optimizer_lrs=get_optimizer_lrs(optimizer),
        optimizer_policy_state=optimizer_policy_state,
        fold_pca_prior=fold_pca_prior,
    )
    release_cuda_memory()
    finish_tag = f"finished:{should_finish_reason}" if should_finish_reason else "max_epoch_or_completed"
    print(f"    [done] {spec.name} Fold{run_idx:02d} {finish_tag}")
    return int(batch_size)


def release_model_level_data_resources(model_name: str, dataset: Any, fold_assignments: Any) -> None:
    """
    释放单个模型全部 Fold 结束后不再需要的数据级资源。
    English: model Fold .

    逻辑说明：
    English: Logic notes:
    1. memory 模式下，SoilMultiSourceDataset.data_cache 会长期持有当前模型的样本张量；
    English: 1. memory , SoilMultiSourceDataset.data_cache currentmodelsample;
    2. 同一模型的多个 Fold 需要复用该缓存，因此不能在 Fold 之间释放；
    English: 2. model Fold cache, Fold ;
    3. 若后续模型数据需求相同，训练主循环会保留并复用 Dataset，不调用本函数；
    English: 3. model, training Dataset, ;
    4. 若后续模型数据需求不同，模型级聚合与 ONNX 导出结束后再清空旧 Dataset，可降低 CPU 内存高水位；
    English: 4. model, model ONNX export Dataset, CPU ;
    5. 该函数只做资源回收，不修改训练参数、折分、缓存策略或输出结果。
    English: 5. , trainingparameter, , cacheOutputresult.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    import gc
    from Train_support import release_cuda_memory

    for attr_name in ("data_cache", "valid_records", "group_records"):
        records = getattr(dataset, attr_name, None)
        if hasattr(records, "clear"):
            records.clear()
    del fold_assignments
    del dataset
    gc.collect()
    release_cuda_memory()
    print(f"[Train_core] {model_name} 模型级数据缓存已释放。")


def run_real_training_loop(menu: ExperimentMenu, output_root: Path) -> None:
    """
    按菜单顺序执行真实训练循环。
    English: menutraining.

    说明 / Notes:
    English: Notes:.
    - 每个模型的 Fold 循环结束后，从磁盘 Fold 输出重建聚合 CSV；
    English: - model Fold , Fold Output CSV;
    - 若入口 Config 启用 ONNX 导出，则在聚合后选择代表 Fold 并导出应用端 ONNX；
    English: - Config ONNX export, select Fold export ONNX;
    - 聚合 CSV 的指标计算放在 Metrics_core，公开版直接保留匿名样本名。
    English: - CSV metriccalculate Metrics_core, public releasesample.
    - 断点续训会先按模型检查未完成 Fold；已完成模型不构建 Dataset，避免重复加载其结构所需数据。
    English: - modelcheck Fold; modelbuild Dataset, avoidload.

    最近修改时间：2026-05-30；作者：ljy。
    English: Last modified: 2026-05-30; Author: ljy.
    """

    from Metrics_core import save_model_aggregate_outputs
    from Train_export_onnx import export_representative_onnx_for_model
    from Train_support import get_split_indices_for_run

    config = menu.Config
    device, device_info = resolve_training_device(config)
    print(f"[Train_core] 训练输出根目录: {output_root}")
    if device_info["training_device_type"] == "cuda":
        print(f"[Train_core] 训练设备: {device_info['training_device']} | {device_info['cuda_device_name']}")
    else:
        print(f"[Train_core] 训练设备: {device_info['training_device']}")

    model_specs = list(menu.MODEL_SPECS)
    retained_dataset = None
    retained_fold_assignments = None
    retained_dataset_key = None
    retained_dataset_owner_name = None

    for order, spec in enumerate(model_specs, start=1):
        print(f"[Train_core] ({order}/{len(menu.MODEL_SPECS)}) 模型: {spec.name}")
        model_dir = output_root / spec.name
        incomplete_run_indices = get_incomplete_run_indices(model_dir, config)
        dataset_key = None
        dataset = None
        fold_assignments = None

        if not incomplete_run_indices:
            print(f"[Train_core] [skip-model] {spec.name} 所有 Fold 已完成，跳过 Dataset 构建与数据加载。")
        else:
            dataset_key = build_dataset_reuse_key_for_spec(spec, config)
            if retained_dataset is not None and retained_dataset_key == dataset_key:
                dataset = retained_dataset
                fold_assignments = retained_fold_assignments
                print(f"[Train_core] {spec.name} 复用上一未完成模型的数据缓存与 Fold 划分。")
            else:
                if retained_dataset is not None:
                    release_model_level_data_resources(
                        model_name=str(retained_dataset_owner_name or "previous_model"),
                        dataset=retained_dataset,
                        fold_assignments=retained_fold_assignments,
                    )
                    retained_dataset = None
                    retained_fold_assignments = None
                    retained_dataset_key = None
                    retained_dataset_owner_name = None
                dataset = build_dataset_for_spec(spec, config)
                fold_assignments = build_fold_assignments_for_training(dataset, config)
                retained_dataset = dataset
                retained_fold_assignments = fold_assignments
                retained_dataset_key = dataset_key
                retained_dataset_owner_name = spec.name

            batch_size, batch_plan = resolve_fold_batch_size(spec, config, model_dir)
            write_auto_batch_plan(model_dir, batch_plan)

            for run_idx in incomplete_run_indices:
                split_indices = get_split_indices_for_run(
                    fold_assignments=fold_assignments,
                    run_idx=run_idx,
                    validation_fold_offset=int(getattr(config, "VALIDATION_FOLD_OFFSET", 1)),
                )
                run_dir = model_dir / f"Fold{run_idx:02d}"
                batch_size = run_one_fold(
                    spec=spec,
                    config=config,
                    dataset=dataset,
                    split_indices=split_indices,
                    run_idx=run_idx,
                    run_dir=run_dir,
                    batch_size=batch_size,
                    device=device,
                )

        written_paths = save_model_aggregate_outputs(
            model_dir=model_dir,
            target_names=get_target_names(config),
            decimals=int(getattr(config, "EXPORT_DECIMALS", 6)),
            sample_name_postprocess_enabled=bool(getattr(config, "SAMPLE_NAME_POSTPROCESS_ENABLED", True)),
        )
        if written_paths:
            print(f"[Train_core] {spec.name} 聚合输出已更新:")
            for path in written_paths.values():
                print(f"  - {path}")
        export_info = export_representative_onnx_for_model(spec=spec, config=config, model_dir=model_dir)
        if export_info:
            print(
                f"[Train_core] {spec.name} 代表 ONNX 已导出: {export_info['onnx_path']} "
                f"(source={export_info['selected_fold']}, {export_info['selection_metric']}="
                f"{float(export_info['selection_metric_value']):.6f})"
            )
            print(f"  - {export_info['export_info_path']}")
        if dataset_key is None:
            continue
        next_spec = find_next_incomplete_spec(
            model_specs=model_specs,
            start_index=order,
            output_root=output_root,
            config=config,
        )
        next_dataset_key = build_dataset_reuse_key_for_spec(next_spec, config) if next_spec is not None else None
        if next_dataset_key == dataset_key:
            print(f"[Train_core] {spec.name} 模型级数据缓存保留，后续未完成模型 {next_spec.name} 继续复用。")
            continue
        release_model_level_data_resources(
            model_name=spec.name,
            dataset=retained_dataset,
            fold_assignments=retained_fold_assignments,
        )
        retained_dataset = None
        retained_fold_assignments = None
        retained_dataset_key = None
        retained_dataset_owner_name = None


def run_training_menu(menu: ExperimentMenu) -> dict[str, Any]:
    """
    共用训练入口。
    English: training.

    说明 / Notes:
    English: Notes:.
    - 当前 V2 先完成工程边界重构，并写出确定的训练计划；
    English: - current V2 , training;
    - 后续正式训练循环只能在本函数内继续实现，禁止新增多个训练执行主体；
    English: - training, training;
    - 正式训练循环必须按菜单 `MODEL_SPECS` 顺序逐个调用 `Model_*.py` 构建未训练模型。
    English: - trainingmenu `MODEL_SPECS` `Model_*.py` buildtrainingmodel.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    menu.sync_to_engine()
    plan = build_training_plan(menu)
    plan_path = write_training_plan(menu, plan)
    print(f"[Train_core] 菜单: {menu.config.name}")
    print(f"[Train_core] 目标变量: {plan['target_names']}")
    print(f"[Train_core] 模型数量: {len(menu.MODEL_SPECS)}")
    print("[Train_core] 执行顺序: 菜单顺序（从小模型到大模型）")
    for index, spec in enumerate(menu.MODEL_SPECS, start=1):
        print(f"  {index}. {spec.name} | {spec.display_name} | batch={spec.batch_size} | image_size={spec.image_size}")
    print(f"[Train_core] 训练计划已写入: {plan_path}")
    output_root = resolve_output_root(menu)
    run_real_training_loop(menu, output_root)
    return plan


def clone_model_spec(spec: ModelSpec, **changes) -> ModelSpec:
    """
    保守复制 ModelSpec，供终端参数层替换 priors_path 等字段使用。
    English: ModelSpec, parameter priors_path field.
    """

    return replace(spec, **changes)


__all__ = [
    "CommonTrainConfig",
    "ExperimentMenu",
    "ModelSpec",
    "TrainingConfig",
    "build_training_plan",
    "clone_model_spec",
    "resolve_training_device",
    "run_training_menu",
    "serialize_model_spec",
    "write_training_plan",
]
