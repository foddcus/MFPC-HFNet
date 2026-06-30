# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
训练终端参数层辅助模块。
Training terminal configuration helpers.

逻辑说明 / Logic
English: Logic.
----------------
1. `Train_main.py` 只负责终端交互和启动，本文件负责菜单注册、参数解析、默认值合并和最终配置摘要。
English: 1. `Train_main.py` , filemenu, parameterparse, defaultconfiguration.
2. 参数优先级固定为：菜单特化设定 > 终端设定 > 训练代码默认。
English: 2. parameterPriority: menu > > trainingdefault.
3. 终端参数没有传入、传入空字符串或交互时直接回车，都表示“不覆盖”，继续使用当前代码默认值。
English: 3. parameter, , “”, currentdefault.
4. 菜单特化设定主要包括模型清单、模型输入结构、分辨率、单模型 batch size、active_inputs、显示名和模型结构超参数；终端层不批量改写这些模型规格。
English: 4. menumodel, modelInput, , model batch size, active_inputs, modelparameter; model.
5. 2026-05-29 起，终端菜单显式支持 CPU 数据缓存内存容量设置；GPU auto-batch/OOM 降批仍由训练引擎层维护。
English: 5. 2026-05-29 , menuexplicit CPU cache; GPU auto-batch/OOM training engine.

最近修改时间 / Last modified: 2026-05-30
English: Last modified: 2026-05-30.
作者 / Author: ljy
English: Author: ljy.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MENU_KEY = "mfpchfnetv2"


@dataclass(frozen=True)
class MenuEntry:
    """
    可由训练终端选择的一条菜单入口。
    English: trainingselectmenu.

    字段说明：
    English: Field notes:
    - `key` 是终端命令使用的短名称；
    English: - `key` name;
    - `module_name` 和 `getter_name` 用于延迟导入，避免只列菜单时提前加载重训练依赖；
    English: - `module_name` `getter_name` , avoidmenuloadtraining;
    - `preset_choices` 记录当前菜单支持的训练清单快捷选择。
    English: - `preset_choices` currentmenutrainingselect.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    key: str
    label: str
    module_name: str
    getter_name: str
    preset_choices: tuple[str, ...]
    default_preset: str = "all"


MENU_REGISTRY: dict[str, MenuEntry] = {
    "mfpchfnetv2": MenuEntry(
        key="mfpchfnetv2",
        label="MFPC-HFNetV2 结构消融 / Full 主模型菜单",
        module_name="Menu_MFPCHFNetV2",
        getter_name="get_mfpchfnetv2_menu",
        preset_choices=("all", "full_only", "ablation_only"),
    ),
    "compare": MenuEntry(
        key="compare",
        label="AllBackbones 全模型 baseline 对比菜单",
        module_name="Menu_Compare_AllBackbones",
        getter_name="get_compare_all_backbones_menu",
        preset_choices=("all", "classic_only", "efficientnet_only"),
    ),
    "input_ablation": MenuEntry(
        key="input_ablation",
        label="输入端消融菜单",
        module_name="Menu_InputAblation",
        getter_name="get_input_ablation_menu",
        preset_choices=("all", "single_only", "pair_only"),
    ),
}


@dataclass(frozen=True)
class ConfigOverrideSpec:
    """
    终端参数到 legacy Config 字段的映射。
    English: parameter legacy Config field.

    说明：
    English: :
    - 该映射只覆盖训练 Config 默认值，不直接改写菜单中的 ModelSpec 结构参数。
    English: - training Config default, menu ModelSpec parameter.
    - 若某个菜单对应的 Config 没有目标字段，则该参数会被跳过并在摘要中提示。
    English: - menu Config field, parameter.

    最近修改时间 / Last modified: 2026-05-24
    English: Last modified: 2026-05-24.
    作者 / Author: ljy
    English: Author: ljy.
    """

    cli_name: str
    attr_name: str
    value_kind: str
    help_text: str
    prompt_label: str
    choices: Optional[tuple[str, ...]] = None
    prompt_group: str = "core"
    menu_keys: Optional[tuple[str, ...]] = None


OVERRIDE_SPECS: tuple[ConfigOverrideSpec, ...] = (
    ConfigOverrideSpec("train_model_preset", "TRAIN_MODEL_PRESET", "str", "训练模型快捷选择。", "训练模型 preset"),
    ConfigOverrideSpec("train_model_names", "TRAIN_MODEL_NAMES", "str_list", "逗号分隔的模型 name；空值表示不手动指定。", "手动模型 name 列表"),
    ConfigOverrideSpec("target_mode", "TARGET_MODE", "str", "目标变量模式。", "目标变量模式", ("soc", "tn", "both")),
    ConfigOverrideSpec("resume_training", "RESUME_TRAINING", "bool", "是否断点续训。", "是否断点续训"),
    ConfigOverrideSpec("resume_save_dir", "RESUME_SAVE_DIR", "path", "断点续训目录；空值使用训练代码默认。", "断点续训目录"),
    ConfigOverrideSpec("dataset_root", "DATASET_ROOT", "path", "公开版匿名单文件数据库根目录；会派生 DATA_DIR=samples。", "数据库根目录"),
    ConfigOverrideSpec("data_dir", "DATA_DIR", "path", "公开版样本 .npz 目录。", "样本文件目录", prompt_group="advanced"),
    ConfigOverrideSpec("gt_path", "GT_PATH", "path", "兼容旧库的 SOC 真值路径；公开版通常留空。", "SOC 真值路径", prompt_group="advanced"),
    ConfigOverrideSpec("tn_path", "TN_PATH", "path", "兼容旧库的 TN 真值路径；公开版通常留空。", "TN 真值路径", prompt_group="advanced"),
    ConfigOverrideSpec("pca_priors_path", "PCA_PRIORS_PATH", "path", "PCA 先验路径；仅在对应菜单支持时生效。", "PCA 先验路径", prompt_group="advanced"),
    ConfigOverrideSpec("full_reference_run_dir", "FULL_REFERENCE_RUN_DIR", "path", "输入端消融 Full 只读参考结果目录。", "Full 参考结果目录", prompt_group="advanced", menu_keys=("input_ablation",)),
    ConfigOverrideSpec("shared_folds_csv_path", "SHARED_FOLDS_CSV_PATH", "path", "外部 shared_folds/fold_assignments.csv 路径。", "shared_folds CSV 路径", prompt_group="advanced"),
    ConfigOverrideSpec("load_shared_folds_from_csv", "LOAD_SHARED_FOLDS_FROM_CSV", "bool", "是否从指定 CSV 读取折分。", "是否读取 shared_folds CSV", prompt_group="advanced"),
    ConfigOverrideSpec("base_run_dir", "BASE_RUN_DIR", "path", "运行根目录；会派生 ModelData。", "运行根目录", prompt_group="advanced"),
    ConfigOverrideSpec("model_data_dir", "MODEL_DATA_DIR", "path", "ModelData 输出根目录。", "ModelData 输出根目录", prompt_group="advanced"),
    ConfigOverrideSpec("cache_mode", "CACHE_MODE", "str", "数据缓存模式：auto/memory/disk。", "数据缓存模式", ("auto", "memory", "disk"), prompt_group="advanced"),
    ConfigOverrideSpec("memory_limit", "MEMORY_LIMIT", "str", "CPU 数据缓存容量上限，例如 128GB。", "CPU 缓存容量上限", prompt_group="advanced"),
    ConfigOverrideSpec("memory_utilization_ratio", "MEMORY_UTILIZATION_RATIO", "float", "CPU 数据缓存容量可用比例。", "CPU 缓存可用比例", prompt_group="advanced"),
    ConfigOverrideSpec("memory_estimate_safety_factor", "MEMORY_ESTIMATE_SAFETY_FACTOR", "float", "数据缓存内存估算安全倍率。", "缓存估算安全倍率", prompt_group="advanced"),
    ConfigOverrideSpec("cache_root", "CACHE_ROOT", "optional_path", "磁盘缓存根目录；空值表示使用默认 registry 策略。", "磁盘缓存根目录", prompt_group="advanced"),
    ConfigOverrideSpec("disk_cache_policy", "DISK_CACHE_POLICY", "str", "磁盘缓存策略。", "磁盘缓存策略", ("reuse_or_build", "reuse_only", "rebuild_all"), prompt_group="advanced"),
    ConfigOverrideSpec("rebuild_preprocess_cache", "REBUILD_PREPROCESS_CACHE", "bool", "是否强制重建预处理缓存。", "重建预处理缓存", prompt_group="advanced"),
    ConfigOverrideSpec("cache_registry_enabled", "CACHE_REGISTRY_ENABLED", "bool", "是否启用磁盘缓存注册表。", "启用缓存注册表", prompt_group="advanced"),
    ConfigOverrideSpec("cache_registry_filename", "CACHE_REGISTRY_FILENAME", "str", "磁盘缓存注册表文件名。", "缓存注册表文件名", prompt_group="advanced"),
    ConfigOverrideSpec("max_epochs", "MAX_EPOCHS", "int", "最大 epoch 数。", "最大 epoch"),
    ConfigOverrideSpec("learning_rate", "LEARNING_RATE", "float", "基础学习率。", "学习率"),
    ConfigOverrideSpec("weight_decay", "WEIGHT_DECAY", "float", "权重衰减。", "权重衰减", prompt_group="advanced"),
    ConfigOverrideSpec("val_interval", "VAL_INTERVAL", "int", "基础验证间隔。", "验证间隔", prompt_group="advanced"),
    ConfigOverrideSpec("num_folds", "NUM_FOLDS", "int", "交叉验证折数。", "交叉验证折数"),
    ConfigOverrideSpec("num_runs", "NUM_RUNS", "int", "实际运行 Fold 数。", "运行 Fold 数", prompt_group="advanced"),
    ConfigOverrideSpec("validation_fold_offset", "VALIDATION_FOLD_OFFSET", "int", "验证折相对测试折偏移。", "验证折偏移"),
    ConfigOverrideSpec("split_seed", "SPLIT_SEED", "int", "稳定折分随机种子。", "折分随机种子"),
    ConfigOverrideSpec("lr_patience_cycles", "LR_PATIENCE_CYCLES", "int", "基础 LR patience 验证单位。", "LR patience", prompt_group="advanced"),
    ConfigOverrideSpec("max_lr_decays", "MAX_LR_DECAYS", "int", "最大学习率衰减次数。", "最大 LR 衰减次数", prompt_group="advanced"),
    ConfigOverrideSpec("lr_decay_factor", "LR_DECAY_FACTOR", "float", "学习率衰减因子。", "LR 衰减因子", prompt_group="advanced"),
    ConfigOverrideSpec("train_loss_patience_bonus_enabled", "TRAIN_LOSS_PATIENCE_BONUS_ENABLED", "bool", "是否启用训练损失刷新临时 patience。", "训练损失临时 patience", prompt_group="advanced"),
    ConfigOverrideSpec("train_loss_patience_bonus_cycles", "TRAIN_LOSS_PATIENCE_BONUS_CYCLES", "int", "训练损失刷新时增加的 patience 单位。", "训练损失 bonus 单位", prompt_group="advanced"),
    ConfigOverrideSpec("train_loss_patience_max_multiplier", "TRAIN_LOSS_PATIENCE_MAX_MULTIPLIER", "float", "训练损失临时 patience 上限倍率。", "训练损失 patience 上限倍率", prompt_group="advanced"),
    ConfigOverrideSpec("lr_patience_grad_accum_extra_multiplier", "LR_PATIENCE_GRAD_ACCUM_EXTRA_MULTIPLIER", "float", "低 batch 梯度累积 patience 额外倍率。", "梯度累积 patience 额外倍率", prompt_group="advanced"),
    ConfigOverrideSpec("min_effective_update_batch_size", "MIN_EFFECTIVE_UPDATE_BATCH_SIZE", "int", "梯度累积目标有效 batch 下限。", "有效 batch 下限", prompt_group="advanced"),
    ConfigOverrideSpec("lr_patience_small_batch_threshold", "LR_PATIENCE_SMALL_BATCH_THRESHOLD", "int", "旧小 batch patience 阈值兼容字段。", "旧小 batch patience 阈值", prompt_group="advanced"),
    ConfigOverrideSpec("lr_patience_small_batch_multiplier", "LR_PATIENCE_SMALL_BATCH_MULTIPLIER", "float", "旧小 batch patience 倍率兼容字段。", "旧小 batch patience 倍率", prompt_group="advanced"),
    ConfigOverrideSpec("lr_patience_batch_one_multiplier", "LR_PATIENCE_BATCH_ONE_MULTIPLIER", "float", "旧 batch=1 patience 倍率兼容字段。", "旧 batch=1 patience 倍率", prompt_group="advanced"),
    ConfigOverrideSpec("freeze_batchnorm_when_batch_lt_min_effective", "FREEZE_BATCHNORM_WHEN_BATCH_LT_MIN_EFFECTIVE", "bool", "小 batch 时是否冻结 BatchNorm running statistics。", "小 batch 冻结 BN", prompt_group="advanced"),
    ConfigOverrideSpec("export_decimals", "EXPORT_DECIMALS", "int", "CSV/汇总导出小数位数。", "导出小数位数", prompt_group="advanced"),
    ConfigOverrideSpec("export_onnx_after_training", "EXPORT_ONNX_AFTER_TRAINING", "bool", "训练完成后是否自动导出每模型代表 ONNX。", "训练后导出 ONNX", prompt_group="advanced"),
    ConfigOverrideSpec("onnx_export_opset", "ONNX_EXPORT_OPSET", "int", "ONNX 导出 opset version。", "ONNX opset", prompt_group="advanced"),
    ConfigOverrideSpec("onnx_export_select_fold_by", "ONNX_EXPORT_SELECT_FOLD_BY", "str", "代表 Fold 选择规则。", "ONNX 代表 Fold 规则", ("test_rmse",), prompt_group="advanced"),
    ConfigOverrideSpec("onnx_export_dynamic_batch", "ONNX_EXPORT_DYNAMIC_BATCH", "bool", "ONNX 是否使用动态 batch 轴。", "ONNX 动态 batch", prompt_group="advanced"),
    ConfigOverrideSpec("onnx_export_name_template", "ONNX_EXPORT_NAME_TEMPLATE", "str", "ONNX 文件名模板。", "ONNX 文件名模板", prompt_group="advanced"),
    ConfigOverrideSpec("nir_dim", "NIR_DIM", "int", "NIR 输入维度。", "NIR 维度", prompt_group="advanced"),
    ConfigOverrideSpec("hyper_dim", "HYPER_DIM", "int", "HyperVISNIR 输入维度。", "Hyper 维度", prompt_group="advanced"),
    ConfigOverrideSpec("image_channels", "IMAGE_CHANNELS", "int", "图像输入通道数。", "图像通道数", prompt_group="advanced"),
    ConfigOverrideSpec("cleanup_completed_fold_checkpoints", "CLEANUP_COMPLETED_FOLD_CHECKPOINTS", "bool", "完成 Fold 后是否清理临时断点。", "清理完成 Fold 临时断点", prompt_group="advanced"),
    ConfigOverrideSpec("keep_completed_fold_best_model", "KEEP_COMPLETED_FOLD_BEST_MODEL", "bool", "清理时是否保留 best_model.pth。", "保留 best_model.pth", prompt_group="advanced"),
)


SPEC_BY_CLI_NAME = {spec.cli_name: spec for spec in OVERRIDE_SPECS}


@dataclass
class AppliedOverride:
    """
    记录一条实际应用或派生得到的参数覆盖。
    English: parameter.
    """

    attr_name: str
    old_value: Any
    new_value: Any
    source: str


@dataclass
class SkippedOverride:
    """
    记录一条被跳过的参数覆盖。
    English: parameter.
    """

    attr_name: str
    raw_value: Any
    reason: str


@dataclass
class PreparedRun:
    """
    终端启动前的最终运行配置。
    English: configuration.

    最近修改时间 / Last modified: 2026-05-24
    English: Last modified: 2026-05-24.
    作者 / Author: ljy
    English: Author: ljy.
    """

    menu_key: str
    menu_entry: MenuEntry
    menu: Any
    applied_overrides: list[AppliedOverride]
    skipped_overrides: list[SkippedOverride]
    config_value_sources: dict[str, str]
    dry_run: bool
    interactive_mode: bool = False
    skip_confirm: bool = False


def normalize_blank(value: Any) -> Any:
    """
    空字符串、纯空白和 None 都视为未传入。
    English: , None .
    """

    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def normalize_path_text(value: Any) -> Optional[str]:
    """
    规范化终端传入路径。
    English: normalizepath.

    规则：
    English: :
    - 空值返回 None，表示不覆盖；
    English: - empty valuereturn None, ;
    - 相对路径按当前 `Python script V2` 目录解释；
    English: - pathcurrent `Python script V2` directory;
    - 不要求路径当前存在，便于新输出目录或未来缓存目录配置。
    English: - pathcurrent, Outputdirectorycachedirectoryconfiguration.
    """

    value = normalize_blank(value)
    if value is None:
        return None

    raw_text = str(value).strip().strip('"').strip("'")
    path = Path(raw_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return os.path.normpath(str(path))


def parse_bool(value: Any) -> bool:
    """
    解析终端布尔值，兼容中英文常用写法。
    English: parse, compatible.
    """

    value = normalize_blank(value)
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("布尔参数为空，不能解析。")

    text = str(value).strip().lower()
    truthy = {"1", "true", "t", "yes", "y", "on", "是", "对", "开启", "启用"}
    falsy = {"0", "false", "f", "no", "n", "off", "否", "不", "关闭", "禁用"}
    if text in truthy:
        return True
    if text in falsy:
        return False
    raise ValueError(f"无法解析布尔值: {value!r}")


def parse_string_list(value: Any) -> list[str]:
    """
    解析逗号或分号分隔的字符串列表。
    English: parselist.
    """

    value = normalize_blank(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).replace(";", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_value(value: Any, value_kind: str) -> Any:
    """
    按字段类型解析终端参数。
    English: fieldparseparameter.
    """

    value = normalize_blank(value)
    if value is None:
        return None

    if value_kind == "str":
        return str(value).strip()
    if value_kind == "str_list":
        return parse_string_list(value)
    if value_kind == "bool":
        return parse_bool(value)
    if value_kind == "int":
        return int(str(value).strip())
    if value_kind == "float":
        return float(str(value).strip())
    if value_kind == "optional_float":
        value = normalize_blank(value)
        if value is None:
            return None
        return float(str(value).strip())
    if value_kind == "path":
        parsed = normalize_path_text(value)
        if parsed is None:
            return None
        return parsed
    if value_kind == "optional_path":
        return normalize_path_text(value)

    raise ValueError(f"未知参数类型: {value_kind}")


def parse_generic_value(raw_value: Any, current_value: Any) -> Any:
    """
    为 `--set-config ATTR=VALUE` 做保守类型推断。
    English: `--set-config ATTR=VALUE` .

    设计说明：
    English: Design note:
    - 优先按当前 Config 字段类型解析，避免把 bool/int/float 全部变成字符串。
    English: - current Config fieldparse, avoid bool/int/float .
    - 当前默认值为 None 时，保留字符串或空值，便于可空路径字段。
    English: - currentdefault None , empty value, pathfield.
    """

    raw_value = normalize_blank(raw_value)
    if raw_value is None:
        return None

    if isinstance(current_value, bool):
        return parse_bool(raw_value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(str(raw_value).strip())
    if isinstance(current_value, float):
        return float(str(raw_value).strip())
    if isinstance(current_value, list):
        return parse_string_list(raw_value)
    if isinstance(current_value, tuple):
        return tuple(parse_string_list(raw_value))
    return str(raw_value).strip()


def format_value(value: Any) -> str:
    """
    将配置值格式化为终端摘要。
    English: configuration.
    """

    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(str(item) for item in value) + ")"
    return str(value)


def get_config_value(config: Any, attr_name: str) -> Any:
    """
    读取 Config 字段，缺失时返回 None。
    English: read Config field, missingreturn None.
    """

    return getattr(config, attr_name, None)


def is_menu_declared_attr(config: Any, attr_name: str) -> bool:
    """
    判断字段是否由菜单 Config 显式规定。
    English: determinefieldmenu Config explicit.

    说明:
    English: :
        只检查当前 Config 类自己的 __dict__；从 CommonTrainConfig 继承来的字段
        English: checkcurrent Config __dict__; CommonTrainConfig field.
        视为训练代码默认值，不算菜单显式设定。
        English: trainingdefault, menuexplicit.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    return str(attr_name) in getattr(config, "__dict__", {})


def build_parser() -> argparse.ArgumentParser:
    """
    创建 Train_main.py 使用的命令行解析器。
    English: create Train_main.py parse.

    最近修改时间 / Last modified: 2026-05-24
    English: Last modified: 2026-05-24.
    作者 / Author: ljy
    English: Author: ljy.
    """

    parser = argparse.ArgumentParser(
        description=(
            "训练终端入口：选择 Menu 接口并覆盖训练默认参数。"
            "未传入或传入空字符串的参数不会覆盖代码默认值。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--menu", choices=tuple(MENU_REGISTRY), default=None, help="选择训练菜单接口。")
    parser.add_argument("--interactive", action="store_true", default=None, help="进入交互式参数确认。")
    parser.add_argument("--prompt-all", action="store_true", default=None, help="交互时显示高级参数。")
    parser.add_argument("--yes", action="store_true", default=None, help="跳过启动前确认。")
    parser.add_argument("--dry-run", action="store_true", default=None, help="只打印最终配置，不启动训练。")
    parser.add_argument("--list-menus", action="store_true", default=None, help="列出可选菜单后退出。")
    parser.add_argument("--list-models", action="store_true", default=None, help="列出所选菜单的模型清单后退出。")
    parser.add_argument(
        "--set-config",
        action="append",
        default=[],
        metavar="ATTR=VALUE",
        help="高级覆盖：直接设置 legacy Config 字段；空 VALUE 表示跳过。",
    )

    for spec in OVERRIDE_SPECS:
        arg_name = "--" + spec.cli_name.replace("_", "-")
        parser.add_argument(arg_name, dest=spec.cli_name, default=None, nargs="?", const="", help=spec.help_text)

    return parser


def print_menu_list() -> None:
    """
    打印可用菜单列表。
    English: menulist.
    """

    print("可选训练菜单：")
    for key, entry in MENU_REGISTRY.items():
        default_mark = " [default]" if key == DEFAULT_MENU_KEY else ""
        print(f"  {key:<15} {entry.label}{default_mark}")


def load_menu(menu_key: str) -> Any:
    """
    延迟导入并返回用户选择的 ExperimentMenu。
    English: returnselect ExperimentMenu.
    """

    entry = MENU_REGISTRY[menu_key]
    module = importlib.import_module(entry.module_name)
    getter = getattr(module, entry.getter_name)
    return getter()


def list_models(menu_key: str) -> None:
    """
    列出所选菜单当前可训练模型规格。
    English: menucurrenttrainingmodel.
    """

    menu = load_menu(menu_key)
    print(f"菜单: {menu_key} | {MENU_REGISTRY[menu_key].label}")
    print("模型清单：")
    for index, spec in enumerate(menu.ALL_MODEL_SPECS, start=1):
        parts = [f"{index:02d}. {getattr(spec, 'name', '')}"]
        display_name = getattr(spec, "display_name", "")
        if display_name:
            parts.append(f"display={display_name}")
        for attr in ("resolution", "image_size", "batch_size", "active_inputs", "ablation_mode"):
            if hasattr(spec, attr):
                parts.append(f"{attr}={getattr(spec, attr)}")
        print(" | ".join(parts))


def prompt_text(label: str, default_value: Any, choices: Optional[Sequence[str]] = None) -> Optional[str]:
    """
    显示一个可空交互提示，直接回车表示保留默认值。
    English: , default.
    """

    default_text = format_value(default_value)
    choice_text = ""
    if choices:
        choice_text = " [" + "/".join(str(item) for item in choices) + "]"
    raw = input(f"{label}{choice_text}，当前默认={default_text}，直接回车保留：").strip()
    return raw if raw else None


def choose_menu_interactively(current_menu_key: Optional[str]) -> str:
    """
    交互式选择训练菜单。
    English: selecttrainingmenu.
    """

    default_key = current_menu_key or DEFAULT_MENU_KEY
    print_menu_list()
    raw = input(f"请选择菜单 key，直接回车使用 {default_key}: ").strip()
    if not raw:
        return default_key
    if raw not in MENU_REGISTRY:
        raise ValueError(f"未知菜单 key: {raw}，可选值为 {sorted(MENU_REGISTRY)}")
    return raw


def collect_interactive_overrides(menu_key: str, menu: Any, prompt_all: bool) -> dict[str, Any]:
    """
    交互式收集终端覆盖参数。
    English: parameter.

    说明：
    English: :
    - 默认只显示最常用的训练、数据库、断点和折分参数；
    English: - defaulttraining, , parameter;
    - `--prompt-all` 才显示高级微调、patience、导出和数据缓存容量；模型结构参数由菜单维护；
    English: - `--prompt-all` , patience, exportcache; modelparametermenu;
    - GPU auto-batch/OOM 属于训练引擎资源策略，不进入交互菜单；
    English: - GPU auto-batch/OOM training engine, menu;
    - 所有提示直接回车都不会覆盖现有代码默认值。
    English: - default.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    config = menu.Config
    entry = MENU_REGISTRY[menu_key]
    raw_values: dict[str, Any] = {}
    print("\n交互式参数设置：直接回车表示采用代码当前默认值。")

    for spec in OVERRIDE_SPECS:
        if spec.menu_keys and menu_key not in spec.menu_keys:
            continue
        if spec.prompt_group == "advanced" and not prompt_all:
            continue
        if not hasattr(config, spec.attr_name):
            continue

        choices = spec.choices
        if spec.cli_name == "train_model_preset":
            choices = entry.preset_choices
        raw = prompt_text(spec.prompt_label, getattr(config, spec.attr_name), choices)
        if raw is not None:
            raw_values[spec.cli_name] = raw

    return raw_values


def collect_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """
    从 argparse 结果中提取非空终端覆盖参数。
    English: argparse resultparameter.
    """

    raw_values: dict[str, Any] = {}
    for spec in OVERRIDE_SPECS:
        raw = normalize_blank(getattr(args, spec.cli_name, None))
        if raw is not None:
            raw_values[spec.cli_name] = raw
    return raw_values


def collect_design_overrides(design_defaults: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    从 Train_main.py 显式参数面板中提取非空覆盖。
    English: Train_main.py explicitparameter.

    规则：
    English: :
    - None 和空字符串表示不覆盖底层训练代码默认值；
    English: - None trainingdefault;
    - 该层先于命令行参数应用，命令行非空参数仍可覆盖这里的设计默认。
    English: - parameter, parameterdefault.

    最近修改时间 / Last modified: 2026-05-24
    English: Last modified: 2026-05-24.
    作者 / Author: ljy
    English: Author: ljy.
    """

    raw_values: dict[str, Any] = {}
    design_defaults = design_defaults or {}
    for spec in OVERRIDE_SPECS:
        raw = normalize_blank(design_defaults.get(spec.cli_name))
        if raw is not None:
            raw_values[spec.cli_name] = raw
    return raw_values


def collect_design_set_config(design_defaults: Optional[dict[str, Any]]) -> list[str]:
    """
    读取 Train_main.py 中预留的高级 `set_config` 覆盖列表。
    English: read Train_main.py `set_config` list.
    """

    design_defaults = design_defaults or {}
    raw_items = design_defaults.get("set_config", [])
    if raw_items is None:
        return []
    if isinstance(raw_items, str):
        raw_items = [raw_items]
    return [str(item) for item in raw_items if normalize_blank(item) is not None]


def pick_bool_setting(cli_value: Optional[bool], design_defaults: Optional[dict[str, Any]], key: str, fallback: bool) -> bool:
    """
    按 命令行 > Train_main 显式默认 > fallback 的顺序选择布尔控制项。
    English: > Train_main explicitdefault > fallback select.
    """

    if cli_value is not None:
        return bool(cli_value)
    design_defaults = design_defaults or {}
    design_value = normalize_blank(design_defaults.get(key))
    if design_value is None:
        return bool(fallback)
    return parse_bool(design_value)


def apply_named_overrides(
    menu_key: str,
    menu: Any,
    raw_values: dict[str, Any],
    raw_sources: Optional[dict[str, str]] = None,
) -> tuple[list[AppliedOverride], list[SkippedOverride], set[str]]:
    """
    应用命名终端参数。
    English: parameter.

    返回：
    English: return:
    - 已应用参数；
    English: - parameter;
    - 被跳过参数；
    English: - parameter;
    - 由终端显式设置过的 Config 字段名集合。
    English: - explicit Config field.
    """

    config = menu.Config
    applied: list[AppliedOverride] = []
    skipped: list[SkippedOverride] = []
    changed_attrs: set[str] = set()

    for cli_name, raw_value in raw_values.items():
        spec = SPEC_BY_CLI_NAME[cli_name]
        if spec.menu_keys and menu_key not in spec.menu_keys:
            skipped.append(SkippedOverride(spec.attr_name, raw_value, f"当前菜单 {menu_key} 不使用该参数。"))
            continue
        if not hasattr(config, spec.attr_name):
            skipped.append(SkippedOverride(spec.attr_name, raw_value, "当前菜单 Config 中没有该字段。"))
            continue

        raw_value = normalize_blank(raw_value)
        if raw_value is None:
            skipped.append(SkippedOverride(spec.attr_name, raw_value, "空值表示不覆盖。"))
            continue

        old_value = getattr(config, spec.attr_name)
        new_value = parse_value(raw_value, spec.value_kind)
        if new_value is None:
            skipped.append(SkippedOverride(spec.attr_name, raw_value, "空值表示不覆盖。"))
            continue
        if spec.choices and str(new_value).lower() not in spec.choices:
            raise ValueError(f"{spec.attr_name}={new_value!r} 不在允许范围 {spec.choices} 内。")

        setattr(config, spec.attr_name, new_value)
        applied.append(AppliedOverride(spec.attr_name, old_value, new_value, (raw_sources or {}).get(cli_name, "terminal")))
        changed_attrs.add(spec.attr_name)

    return applied, skipped, changed_attrs


def apply_generic_overrides(menu: Any, generic_items: Iterable[str]) -> tuple[list[AppliedOverride], list[SkippedOverride], set[str]]:
    """
    应用 `--set-config ATTR=VALUE` 覆盖。
    English: `--set-config ATTR=VALUE` .

    该接口用于临时覆盖未显式列出的 Config 字段；仍不允许新增不存在的字段，避免误拼写静默生效。
    English: explicit Config field; field, avoid.
    """

    config = menu.Config
    applied: list[AppliedOverride] = []
    skipped: list[SkippedOverride] = []
    changed_attrs: set[str] = set()

    for item in generic_items:
        if "=" not in str(item):
            raise ValueError(f"--set-config 需要 ATTR=VALUE 格式: {item!r}")
        attr_name, raw_value = str(item).split("=", 1)
        attr_name = attr_name.strip()
        raw_value = normalize_blank(raw_value)
        if not attr_name:
            raise ValueError(f"--set-config 字段名为空: {item!r}")
        if not hasattr(config, attr_name):
            raise ValueError(f"Config 中不存在字段 {attr_name!r}，请检查拼写。")
        if raw_value is None:
            skipped.append(SkippedOverride(attr_name, raw_value, "空值表示不覆盖。"))
            continue

        old_value = getattr(config, attr_name)
        new_value = parse_generic_value(raw_value, old_value)
        if new_value is None:
            skipped.append(SkippedOverride(attr_name, raw_value, "空值表示不覆盖。"))
            continue
        setattr(config, attr_name, new_value)
        applied.append(AppliedOverride(attr_name, old_value, new_value, "terminal:set-config"))
        changed_attrs.add(attr_name)

    return applied, skipped, changed_attrs


def apply_dependent_defaults(menu: Any, changed_attrs: set[str], force_derived_attrs: Optional[set[str]] = None) -> list[AppliedOverride]:
    """
    根据终端覆盖补齐依赖路径。
    English: path.

    例：
    English: :
    - 终端覆盖 `DATASET_ROOT` 后，若没有单独覆盖 `DATA_DIR`/`GT_PATH`/`TN_PATH`，则按公开版数据库根目录派生；
    English: - `DATASET_ROOT` , `DATA_DIR`/`GT_PATH`/`TN_PATH`, public releasedatabase root directory;
    - 终端覆盖 `FULL_REFERENCE_RUN_DIR` 后，若没有单独覆盖 `SHARED_FOLDS_CSV_PATH`，则同步指向对应 shared_folds。
    English: - `FULL_REFERENCE_RUN_DIR` , `SHARED_FOLDS_CSV_PATH`, shared_folds.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    维护记录 / Maintenance:
    English: Maintenance:.
    - 2026-06-16；作者：GG。当命令行/交互层覆盖上级路径时，允许同步刷新 Train_main 面板补充的派生路径。
    English: - 2026-06-16; Author: GG./path, Train_main path.
    """

    config = menu.Config
    applied: list[AppliedOverride] = []
    force_derived_attrs = set(force_derived_attrs or set())

    def maybe_set(attr_name: str, value: Any, source: str) -> None:
        """
        仅在用户未显式覆盖时写入派生默认值。
        English: explicitwritedefault.

        输入:
        English: Input:
            attr_name: 待写入的 Config 属性名。
            English: attr_name: write Config .
            value: 根据上级路径推导出的默认值。
            English: value: pathexportdefault.
            source: 覆盖来源说明，用于终端摘要和日志溯源。
            English: source: , .

        设计说明:
        English: Design note:
            这里严格遵守“终端/用户显式值优先”，不会覆盖 changed_attrs 中已有的用户设置。
            English: “/explicit”, changed_attrs .
            当命令行/交互层改了上级路径而 Train_main 面板保留旧派生路径时，可通过 force_derived_attrs 定向刷新。
            English: /path Train_main path, force_derived_attrs .
            最近修改时间：2026-06-16；作者：GG。
            English: Last modified: 2026-06-16; Author: GG.
        """
        if not hasattr(config, attr_name):
            return
        if attr_name in changed_attrs and attr_name not in force_derived_attrs:
            return
        old_value = getattr(config, attr_name)
        setattr(config, attr_name, value)
        applied.append(AppliedOverride(attr_name, old_value, value, source))

    if "DATASET_ROOT" in changed_attrs and hasattr(config, "DATASET_ROOT"):
        dataset_root = Path(str(config.DATASET_ROOT))
        maybe_set("DATA_DIR", os.path.normpath(str(dataset_root / "samples")), "terminal-derived:dataset_root")
        maybe_set("GT_PATH", "", "terminal-derived:dataset_root")
        maybe_set("TN_PATH", "", "terminal-derived:dataset_root")

    if "BASE_RUN_DIR" in changed_attrs and hasattr(config, "BASE_RUN_DIR"):
        base_run_dir = Path(str(config.BASE_RUN_DIR))
        maybe_set("MODEL_DATA_DIR", os.path.normpath(str(base_run_dir / "ModelData")), "terminal-derived:base_run_dir")

    if "FULL_REFERENCE_RUN_DIR" in changed_attrs and hasattr(config, "FULL_REFERENCE_RUN_DIR"):
        full_reference_dir = Path(str(config.FULL_REFERENCE_RUN_DIR))
        maybe_set(
            "SHARED_FOLDS_CSV_PATH",
            os.path.normpath(str(full_reference_dir / "shared_folds" / "fold_assignments.csv")),
            "terminal-derived:full_reference_run_dir",
        )

    return applied


def refresh_specs_using_terminal_defaults(menu: Any, changed_attrs: set[str], previous_values: dict[str, Any]) -> list[AppliedOverride]:
    """
    在不破坏菜单特化规格的前提下，同步少数“由 Config 派生”的模型字段。
    English: menu, “ Config ”modelfield.

    当前只处理 `PCA_PRIORS_PATH -> ModelSpec.priors_path`：
    English: current `PCA_PRIORS_PATH -> ModelSpec.priors_path`:
    - 若某个 spec 的 priors_path 等于终端覆盖前的 Config.PCA_PRIORS_PATH，说明它只是引用了默认先验路径，可随终端默认值一起更新；
    English: - spec priors_path Config.PCA_PRIORS_PATH, defaultpath, defaultupdate;
    - 若 spec.priors_path 与旧默认不同，视为菜单特化设定，保持不变。
    English: - spec.priors_path default, menu, .
    """

    if "PCA_PRIORS_PATH" not in changed_attrs or not hasattr(menu.Config, "PCA_PRIORS_PATH"):
        return []

    old_default = previous_values.get("PCA_PRIORS_PATH")
    new_default = menu.Config.PCA_PRIORS_PATH
    if old_default is None or str(old_default) == str(new_default):
        return []

    applied: list[AppliedOverride] = []
    new_specs = []
    changed_any = False
    for spec in menu.all_model_specs:
        if hasattr(spec, "priors_path") and str(getattr(spec, "priors_path")) == str(old_default):
            new_spec = replace(spec, priors_path=new_default)
            new_specs.append(new_spec)
            applied.append(
                AppliedOverride(
                    f"ModelSpec({getattr(spec, 'name', '')}).priors_path",
                    old_default,
                    new_default,
                    "terminal-derived:pca_priors_path",
                )
            )
            changed_any = True
        else:
            new_specs.append(spec)

    if changed_any:
        menu.all_model_specs = list(new_specs)
    return applied


def snapshot_selected_config(config: Any) -> dict[str, Any]:
    """
    记录终端层会触碰的关键 Config 默认值。
    English: Config default.
    """

    attrs = {spec.attr_name for spec in OVERRIDE_SPECS}
    return {attr: getattr(config, attr) for attr in attrs if hasattr(config, attr)}


def build_config_value_sources(config: Any, explicit_attrs: set[str], applied: Iterable[AppliedOverride]) -> dict[str, str]:
    """
    构建最终 Config 字段来源摘要。
    English: build Config field.

    来源层级:
    English: :
        1. 已应用覆盖记录中的 source；
        English: 1. source;
        2. 菜单 Config.__dict__ 显式字段；
        English: 2. menu Config.__dict__ explicitfield;
        3. 训练代码默认值。
        English: 3. trainingdefault.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    sources = {
        attr_name: ("菜单显式" if attr_name in explicit_attrs else "core默认")
        for attr_name in snapshot_selected_config(config)
    }
    for item in applied:
        sources[str(item.attr_name)] = str(item.source)
    return sources


def prepare_run(argv: Optional[Sequence[str]] = None, design_defaults: Optional[dict[str, Any]] = None) -> PreparedRun:
    """
    解析终端参数并准备最终训练运行。
    English: parseparametertraining.

    注意：
    English: :
        - 本函数只合并配置，不启动训练；
        English: - configuration, training;
        - 2026-05-29 / ljy：显式记录 Config 字段来源，区分菜单显式、
        English: ljy：显式记录 Config 字段来源，区分菜单显式、.
          Train_main 补充和命令行覆盖，便于 dry-run 核查参数边界。
          English: Train_main , dry-run parameter.
        - 2026-06-16 / GG：命令行或交互覆盖 DATASET_ROOT 时，同步刷新未显式覆盖的 DATA_DIR。
        English: GG：命令行或交互覆盖 DATASET_ROOT 时，同步刷新未显式覆盖的 DATA_DIR.
    """

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    list_menus = pick_bool_setting(args.list_menus, design_defaults, "list_menus", False)
    if list_menus:
        print_menu_list()
        raise SystemExit(0)

    interactive_fallback = len(argv) == 0
    interactive_mode = pick_bool_setting(args.interactive, design_defaults, "interactive", interactive_fallback)
    menu_key = args.menu or normalize_blank((design_defaults or {}).get("menu"))
    if interactive_mode:
        menu_key = choose_menu_interactively(menu_key)
    if menu_key is None:
        menu_key = DEFAULT_MENU_KEY

    list_models_requested = pick_bool_setting(args.list_models, design_defaults, "list_models", False)
    if list_models_requested:
        list_models(menu_key)
        raise SystemExit(0)

    menu_entry = MENU_REGISTRY[menu_key]
    menu = load_menu(menu_key)
    previous_values = snapshot_selected_config(menu.Config)
    explicit_config_attrs = set(getattr(menu.Config, "__dict__", {}))

    design_values = collect_design_overrides(design_defaults)
    cli_values = collect_cli_overrides(args)
    raw_values = dict(design_values)
    raw_sources = {key: "Train_main补充" for key in design_values}
    raw_values.update(cli_values)
    raw_sources.update({key: "命令行覆盖" for key in cli_values})
    prompt_all = pick_bool_setting(args.prompt_all, design_defaults, "prompt_all", False)
    interactive_values: dict[str, Any] = {}
    if interactive_mode:
        interactive_values = collect_interactive_overrides(menu_key, menu, prompt_all)
        raw_values.update(interactive_values)
        raw_sources.update({key: "交互覆盖" for key in interactive_values})

    applied_named, skipped_named, changed_named = apply_named_overrides(menu_key, menu, raw_values, raw_sources)
    set_config_items = collect_design_set_config(design_defaults) + list(args.set_config)
    applied_generic, skipped_generic, changed_generic = apply_generic_overrides(menu, set_config_items)
    changed_attrs = set(changed_named) | set(changed_generic)
    runtime_value_keys = set(cli_values) | set(interactive_values)
    runtime_changed_attrs = {
        SPEC_BY_CLI_NAME[key].attr_name for key in runtime_value_keys if key in SPEC_BY_CLI_NAME
    } | set(changed_generic)
    force_derived_attrs: set[str] = set()
    if "DATASET_ROOT" in runtime_changed_attrs:
        force_derived_attrs.update({"DATA_DIR", "GT_PATH", "TN_PATH"} - runtime_changed_attrs)
    if "BASE_RUN_DIR" in runtime_changed_attrs:
        force_derived_attrs.update({"MODEL_DATA_DIR"} - runtime_changed_attrs)
    if "FULL_REFERENCE_RUN_DIR" in runtime_changed_attrs:
        force_derived_attrs.update({"SHARED_FOLDS_CSV_PATH"} - runtime_changed_attrs)

    applied_dependent = apply_dependent_defaults(menu, changed_attrs, force_derived_attrs)
    applied_spec_defaults = refresh_specs_using_terminal_defaults(menu, changed_attrs, previous_values)
    applied_all = applied_named + applied_generic + applied_dependent + applied_spec_defaults

    # 最后同步菜单清单。该顺序体现“菜单特化设定 > 终端设定 > 训练代码默认”：
    # EN: most after same single single. this order order " > > ":
    # 终端先改 Config 默认值，菜单再把模型规格清单覆盖回 legacy 主体。
    # EN: terminal first change Config default value, single then model specification single override legacy.
    menu.sync_to_engine()

    dry_run = pick_bool_setting(args.dry_run, design_defaults, "dry_run", False)
    skip_confirm = pick_bool_setting(args.yes, design_defaults, "yes", False)

    return PreparedRun(
        menu_key=menu_key,
        menu_entry=menu_entry,
        menu=menu,
        applied_overrides=applied_all,
        skipped_overrides=skipped_named + skipped_generic,
        config_value_sources=build_config_value_sources(menu.Config, explicit_config_attrs, applied_all),
        dry_run=dry_run,
        interactive_mode=interactive_mode,
        skip_confirm=skip_confirm,
    )


def summarize_run(prepared: PreparedRun) -> str:
    """
    生成最终训练配置摘要。
    English: trainingconfiguration.

    最近修改时间：2026-05-24；作者：ljy。
    English: Last modified: 2026-05-24; Author: ljy.
    """

    config = prepared.menu.Config
    model_specs = list(prepared.menu.MODEL_SPECS)
    lines: list[str] = []
    lines.append("最终训练配置摘要")
    lines.append("=" * 60)
    lines.append(f"菜单接口: {prepared.menu_key} | {prepared.menu_entry.label}")
    lines.append(f"参数优先级: 菜单显式设定 > Train_main 面板补充 > 训练代码默认；命令行显式参数用于临时覆盖。")
    lines.append(f"训练模型数量: {len(model_specs)}")
    lines.append("训练模型:")
    for spec in model_specs:
        detail_parts = [str(getattr(spec, "name", ""))]
        for attr in (
            "resolution",
            "image_size",
            "batch_size",
            "active_inputs",
            "ablation_mode",
            "optimizer_policy",
            "backbone_lr",
            "head_lr",
            "freeze_backbone_epochs",
            "disable_dropout_droppath",
        ):
            if hasattr(spec, attr):
                detail_parts.append(f"{attr}={getattr(spec, attr)}")
        lines.append("  - " + " | ".join(detail_parts))

    key_attrs = [
        "TRAIN_MODEL_PRESET",
        "TRAIN_MODEL_NAMES",
        "TARGET_MODE",
        "RESUME_TRAINING",
        "RESUME_SAVE_DIR",
        "DATASET_ROOT",
        "DATA_DIR",
        "GT_PATH",
        "TN_PATH",
        "PCA_PRIORS_PATH",
        "FULL_REFERENCE_RUN_DIR",
        "SHARED_FOLDS_CSV_PATH",
        "CACHE_MODE",
        "MEMORY_LIMIT",
        "MEMORY_UTILIZATION_RATIO",
        "MEMORY_ESTIMATE_SAFETY_FACTOR",
        "CACHE_ROOT",
        "DISK_CACHE_POLICY",
        "REBUILD_PREPROCESS_CACHE",
        "CACHE_REGISTRY_ENABLED",
        "CACHE_REGISTRY_FILENAME",
        "MAX_EPOCHS",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "VAL_INTERVAL",
        "LR_PATIENCE_CYCLES",
        "MAX_LR_DECAYS",
        "LR_DECAY_FACTOR",
        "TRAIN_LOSS_PATIENCE_BONUS_ENABLED",
        "TRAIN_LOSS_PATIENCE_BONUS_CYCLES",
        "TRAIN_LOSS_PATIENCE_MAX_MULTIPLIER",
        "LR_PATIENCE_GRAD_ACCUM_EXTRA_MULTIPLIER",
        "MIN_EFFECTIVE_UPDATE_BATCH_SIZE",
        "LR_PATIENCE_SMALL_BATCH_THRESHOLD",
        "LR_PATIENCE_SMALL_BATCH_MULTIPLIER",
        "LR_PATIENCE_BATCH_ONE_MULTIPLIER",
        "FREEZE_BATCHNORM_WHEN_BATCH_LT_MIN_EFFECTIVE",
        "EXPORT_DECIMALS",
        "EXPORT_ONNX_AFTER_TRAINING",
        "ONNX_EXPORT_OPSET",
        "ONNX_EXPORT_SELECT_FOLD_BY",
        "ONNX_EXPORT_DYNAMIC_BATCH",
        "ONNX_EXPORT_NAME_TEMPLATE",
        "EVAL_BATCH_SIZE",
        "NUM_FOLDS",
        "NUM_RUNS",
        "VALIDATION_FOLD_OFFSET",
        "SPLIT_SEED",
        "MODEL_DATA_DIR",
        "SAVE_DIR",
    ]
    lines.append("")
    lines.append("关键 Config:")
    for attr in key_attrs:
        if hasattr(config, attr):
            source = prepared.config_value_sources.get(attr, "未知来源")
            lines.append(f"  {attr}: {format_value(getattr(config, attr))} [{source}]")

    lines.append("")
    if prepared.applied_overrides:
        lines.append("终端覆盖/派生参数:")
        for item in prepared.applied_overrides:
            lines.append(
                f"  - {item.attr_name}: {format_value(item.old_value)} -> "
                f"{format_value(item.new_value)} [{item.source}]"
            )
    else:
        lines.append("终端覆盖/派生参数: 无，全部采用菜单与训练代码默认值。")

    if prepared.skipped_overrides:
        lines.append("")
        lines.append("跳过的终端参数:")
        for item in prepared.skipped_overrides:
            lines.append(f"  - {item.attr_name}: {item.reason}")

    return "\n".join(lines)


