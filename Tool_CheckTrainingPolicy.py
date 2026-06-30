# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
V2 训练工程边界静态检查脚本。
Static checker for V2 training-code boundaries.

逻辑 / Logic
English: Logic.
------------
1. 检查 V2 是否符合“菜单 / 主管 / 厨师 / 食材 / 调料 / 上菜员”的工程设计。
English: 1. check V2 “menu / / / / / ”.
2. 禁止出现多个训练执行主体，防止训练边界再次分裂。
English: 2. training, training.
3. 检查菜单声明 `ModelSpec`、特殊训练参数和执行顺序，不承载训练循环、数据加载或结果导出。
English: 3. checkmenu `ModelSpec`, trainingparameter, training, loadresultexport.
4. 检查指标与输出工具集中在 `Metrics_` 前缀文件。
English: 4. checkmetricOutput `Metrics_` file.
5. 检查 optimizer 策略按“菜单声明、模型分组、Train_optimizer 执行”的边界复用。
English: 5. check optimizer policy“menu, model, Train_optimizer ”.
6. 检查 Train_main 参数补充层不得覆盖菜单显式设定，并禁止 main/core 暴露 FFN_RATIO 全局入口。
English: 6. check Train_main parametermenuexplicit, main/core FFN_RATIO .
7. 检查训练完成后的应用端 ONNX 导出保持在 Train_export_onnx.py 独立模块。
English: 7. checktraining ONNX export Train_export_onnx.py .
8. 检查 MFPC-HFNet 结构尺寸和结构标签只能由菜单显式声明。
English: 8. check MFPC-HFNet labelmenuexplicit.

最近修改时间 / Last modified: 2026-05-30
English: Last modified: 2026-05-30.
作者 / Author: ljy

Usage:
    python Tool_CheckTrainingPolicy.py
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = {
    "Train_main.py",
    "Train_config.py",
    "Train_core.py",
    "Train_export_onnx.py",
    "Train_optimizer.py",
    "Train_support.py",
    "Metrics_core.py",
    "Menu_MFPCHFNetV2.py",
    "Menu_InputAblation.py",
    "Menu_Compare_AllBackbones.py",
    "Model_CompareBackbones.py",
    "Model_MFPCHFNet.py",
    "Model_EfficientNet1024Backbones.py",
    "Data_LoaderRuntimeAuto.py",
    "Data_DiskCacheRegistry.py",
    "Data_BuildPcaPriorsFull.py",
}

MENU_FILES = (
    "Menu_MFPCHFNetV2.py",
    "Menu_InputAblation.py",
    "Menu_Compare_AllBackbones.py",
)

FORBIDDEN_GLOBAL_TOKENS = (
    "runtime" + "_module",
    "Training" + "Work" + "flow",
    "Train" + "Runtime_",
    "Train" + "Tool_CompareAllBackbones" + "Pro" + chr(118) + "ider",
    "spec_from_file" + "_location",
    "History" + "Train Code",
    "History_" + "Train Code",
)

MENU_FORBIDDEN_TOKENS = (
    "SoilMultiSourceDataset(",
    "DataLoader(",
    "def run_one_model_one_run",
    "def run_one_model_all_runs",
    "metrics_summary.csv",
    "test_predictions_fold",
)

MFPCHF_STRUCTURE_BOUNDARY_TOKENS = (
    "FULL_IMAGE_SIZE",
    "H2H3LOW_IMAGE_SIZE",
    "H3LOW_IMAGE_SIZE",
    "LOWONLY_IMAGE_SIZE",
    "SHARED_IMAGE_EMBED_DIM",
    "SHARED_HYPER_EMBED_DIM",
    "SHARED_NIR_EMBED_DIM",
    "SHARED_FUSION_HIDDEN_DIM",
    "INPUT_FEATURE_VECTOR_RATIO",
    "ALLOCATION_SOURCE",
    "FREEZE_PCA",
    "MFPCHF_TOKEN_COMPRESSION_RATIO",
    "MFPCHF_TOKEN_DIM_MIN",
    "MFPCHF_TOKEN_DIM_ROUND_MULTIPLE",
    "MFPCHF_LD_ATTN_DIM",
    "MFPCHF_LD_HEADS",
    "MFPCHF_CPE_ATTN_DIM",
    "MFPCHF_CPE_HEADS",
    "MFPCHF_DROPOUT",
)

MFPCHF_IMAGE_SIZE_TOKENS = (
    "FULL_IMAGE_SIZE",
    "H2H3LOW_IMAGE_SIZE",
    "H3LOW_IMAGE_SIZE",
    "LOWONLY_IMAGE_SIZE",
)


@dataclass
class PolicyIssue:
    """
    单条静态检查问题。
    English: A single static-check issue.
    """

    file: str
    line: int
    message: str


def path_of(file_name: str) -> Path:
    """
    返回 V2 源码文件路径。
    English: Return the path of a V2 source file.
    """

    return ROOT / file_name


def read_source(file_name: str) -> str:
    """
    读取 UTF-8 源码。
    English: Read UTF-8 source code.
    """

    return path_of(file_name).read_text(encoding="utf-8-sig")


def line_of(source: str, token: str) -> int:
    """
    返回 token 首次出现的 1-based 行号。
    English: Return the 1-based line number of the first token occurrence.
    """

    before = source.split(token, 1)[0]
    return before.count("\n") + 1


def parse_source(file_name: str) -> ast.Module:
    """
    把源码解析为 AST，用于静态检查显式声明边界。
    English: Parse source code into an AST for static boundary checks.
    """

    return ast.parse(read_source(file_name), filename=file_name)


def class_assignment_names(file_name: str, class_name: str) -> set[str]:
    """
    返回指定类体内直接声明的类属性名称。
    English: Return class attribute names declared directly in the specified class body.
    """

    tree = parse_source(file_name)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            names: set[str] = set()
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
                elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    names.add(statement.target.id)
            return names
    return set()


def source_has_assignment(source: str, name: str) -> bool:
    """
    检查源码中是否存在形如 NAME = value 或 NAME: type = value 的显式赋值。
    English: Check whether source code contains an explicit assignment like `NAME = value` or `NAME: type = value`.
    """

    pattern = rf"(?m)^\s*{re.escape(name)}\s*(?::[^=\n]+)?="
    return re.search(pattern, source) is not None


def literal_string(node: ast.AST | None) -> str:
    """
    读取 AST 字符串常量；无法解析时返回空字符串。
    English: Read an AST string constant; return an empty string when it cannot be parsed.
    """

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def literal_string_tuple(node: ast.AST | None) -> tuple[str, ...]:
    """
    读取 AST 字符串 tuple/list 常量；无法解析时返回空 tuple。
    English: Read an AST string tuple/list constant; return an empty tuple when it cannot be parsed.
    """

    if isinstance(node, (ast.Tuple, ast.List)):
        values = []
        for item in node.elts:
            text = literal_string(item).strip().lower()
            if text:
                values.append(text)
        return tuple(values)
    return ()


def check_required_files() -> list[PolicyIssue]:
    """
    检查 V2 必需文件是否存在。
    English: Check whether required V2 files exist.
    """

    issues: list[PolicyIssue] = []
    for file_name in sorted(REQUIRED_FILES):
        if not path_of(file_name).is_file():
            issues.append(PolicyIssue(file_name, 1, "缺少 V2 工程边界所需文件。"))
    return issues


def check_no_forbidden_execution_split() -> list[PolicyIssue]:
    """
    禁止多个训练执行主体。
    English: Forbid multiple training execution bodies.
    """

    issues: list[PolicyIssue] = []
    for path in sorted(ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        if "work" + "flow" in path.name.lower():
            issues.append(PolicyIssue(path.name, 1, "V2 禁止使用分裂训练执行主体文件。"))
        for token in FORBIDDEN_GLOBAL_TOKENS:
            if token in source:
                issues.append(PolicyIssue(path.name, line_of(source, token), f"V2 禁止出现执行边界分裂标记: {token!r}"))
    return issues


def check_train_core_boundary() -> list[PolicyIssue]:
    """
    检查唯一训练引擎边界。
    English: Check the single training-engine boundary.
    """

    source = read_source("Train_core.py")
    issues: list[PolicyIssue] = []
    required = (
        "class ExperimentMenu",
        "class ModelSpec",
        "def run_training_menu",
        "def run_real_training_loop",
        "def build_model_from_spec",
        "def build_training_plan",
        "V2 禁止再拆出多个训练执行主体",
        "按菜单 `MODEL_SPECS` 顺序逐个调用 `Model_*.py`",
        "Train_optimizer.py",
        "execution_order_policy",
        "menu_order",
        "指标与结果汇总只属于 `Metrics_*.py`",
    )
    for token in required:
        if token not in source:
            issues.append(PolicyIssue("Train_core.py", 1, f"Train_core 缺少唯一训练引擎边界标记: {token}"))
    return issues


def check_optimizer_policy_boundary() -> list[PolicyIssue]:
    """
    检查 optimizer 策略的泛化边界。
    English: Check the generalized optimizer-policy boundary.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    issues: list[PolicyIssue] = []
    optimizer_source = read_source("Train_optimizer.py")
    for token in (
        "def build_optimizer_for_spec",
        "def sync_optimizer_policy_for_epoch",
        "freeze_then_layerwise",
        "layerwise_lr",
        "serialize_optimizer_plan",
    ):
        if token not in optimizer_source:
            issues.append(PolicyIssue("Train_optimizer.py", 1, f"optimizer 策略层缺少通用能力: {token}"))

    model_source = read_source("Model_CompareBackbones.py")
    for token in ("def get_optimizer_parameter_groups", '"lr_role": "backbone"', '"lr_role": "head"'):
        if token not in model_source:
            issues.append(PolicyIssue("Model_CompareBackbones.py", 1, f"Compare 模型缺少参数分组语义: {token}"))

    mfpchf_source = read_source("Model_MFPCHFNet.py")
    for token in ("def split_transformer_fusion_optimizer_groups", '"lr_role": "transformer_fusion"', "image_branch.hlaf.final_cpe"):
        if token not in mfpchf_source:
            issues.append(PolicyIssue("Model_MFPCHFNet.py", 1, f"MFPC-HFNet 模型缺少融合层 optimizer 分组语义: {token}"))

    mfpchf_menu_source = read_source("Menu_MFPCHFNetV2.py")
    for token in ("optimizer_policy=\"layerwise_lr\"", '"transformer_fusion": 0.1', "MFPCHF_TRANSFORMER_FUSION_OPTIMIZER_EXTRA"):
        if token not in mfpchf_menu_source:
            issues.append(PolicyIssue("Menu_MFPCHFNetV2.py", 1, f"MFPC-HFNet 融合层低 LR 菜单声明缺失: {token}"))

    menu_source = read_source("Menu_Compare_AllBackbones.py")
    for token in ("optimizer_policy=\"freeze_then_layerwise\"", "backbone_lr=1e-5", "head_lr=1e-4", "freeze_backbone_epochs=30"):
        if token not in menu_source:
            issues.append(PolicyIssue("Menu_Compare_AllBackbones.py", 1, f"SwinV2 微调策略菜单声明缺失: {token}"))
    return issues


def check_train_main_config_boundary() -> list[PolicyIssue]:
    """
    检查 Train_main / Train_config / Train_core 参数边界。
    English: Check parameter boundaries across Train_main, Train_config, and Train_core.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    issues: list[PolicyIssue] = []
    forbidden = ("MFPCHF_FFN_RATIO", "mfpchf_ffn_ratio")
    for file_name in ("Train_main.py", "Train_config.py", "Train_core.py"):
        source = read_source(file_name)
        for token in forbidden:
            if token in source:
                issues.append(PolicyIssue(file_name, line_of(source, token), f"main/core Config 不得暴露 MFPC-HFNet FFN_RATIO 入口: {token}"))

    train_main_source = read_source("Train_main.py")
    for token in (
        "TRAIN_DATA_SETTING_KEYS",
        "TRAIN_PATIENCE_SETTING_KEYS",
        "TRAIN_OUTPUT_SETTING_KEYS",
        "TRAIN_MODEL_CONFIG_SETTING_KEYS",
        "TRAIN_MAIN_OWNED_SETTING_KEYS",
        "def _menu_declares_config_field",
        "allow_menu_declared=cli_name in TRAIN_MAIN_OWNED_SETTING_KEYS",
    ):
        if token not in train_main_source:
            issues.append(PolicyIssue("Train_main.py", 1, f"Train_main 缺少参数补充层边界标记: {token}"))

    train_config_source = read_source("Train_config.py")
    for token in (
        "config_value_sources",
        "def build_config_value_sources",
        "Train_main补充",
        "命令行覆盖",
        "菜单显式",
        "core默认",
        "LR_DECAY_FACTOR",
        "TRAIN_LOSS_PATIENCE_BONUS_CYCLES",
        "EXPORT_DECIMALS",
    ):
        if token not in train_config_source:
            issues.append(PolicyIssue("Train_config.py", 1, f"Train_config 缺少参数来源摘要或关键字段: {token}"))
    return issues


def check_mfpchf_structure_boundary() -> list[PolicyIssue]:
    """
    检查 MFPC-HFNet 结构参数只由菜单显式声明。
    English: Check that MFPC-HFNet architecture parameters are declared only by menus.

    最近修改时间：2026-05-30；作者：ljy。
    English: Last modified: 2026-05-30; Author: ljy.
    """

    issues: list[PolicyIssue] = []

    for file_name in ("Train_main.py", "Train_config.py"):
        source = read_source(file_name)
        for token in MFPCHF_STRUCTURE_BOUNDARY_TOKENS:
            if source_has_assignment(source, token):
                issues.append(PolicyIssue(file_name, line_of(source, token), f"MFPC-HFNet 结构参数不得在入口/配置层赋值: {token}"))

    common_assignments = class_assignment_names("Train_core.py", "CommonTrainConfig")
    for token in MFPCHF_STRUCTURE_BOUNDARY_TOKENS:
        if token in common_assignments:
            issues.append(PolicyIssue("Train_core.py", line_of(read_source("Train_core.py"), token), f"CommonTrainConfig 不得保留 MFPC-HFNet 结构默认值: {token}"))

    for file_name in ("Menu_MFPCHFNetV2.py", "Menu_InputAblation.py"):
        menu_assignments = class_assignment_names(file_name, "Config")
        for token in MFPCHF_IMAGE_SIZE_TOKENS:
            if token not in menu_assignments:
                issues.append(PolicyIssue(file_name, 1, f"菜单 Config 必须显式声明 MFPC-HFNet 结构尺寸: {token}"))

    mfpchf_menu = parse_source("Menu_MFPCHFNetV2.py")
    for node in ast.walk(mfpchf_menu):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "ModelSpec":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        if literal_string(keywords.get("model_family")) != "mfpchfnet":
            continue
        model_name = literal_string(keywords.get("name")) or "MFPC-HFNet ModelSpec"
        if not literal_string(keywords.get("structure")).strip():
            issues.append(PolicyIssue("Menu_MFPCHFNetV2.py", getattr(node, "lineno", 1), f"{model_name} 必须显式声明 structure。"))

    input_menu = parse_source("Menu_InputAblation.py")
    for node in ast.walk(input_menu):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "ModelSpec":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        if literal_string(keywords.get("model_family")) != "input_ablation":
            continue
        model_name = literal_string(keywords.get("name")) or "input_ablation ModelSpec"
        active_inputs = literal_string_tuple(keywords.get("active_inputs"))
        structure = literal_string(keywords.get("structure")).strip()
        if "image" in active_inputs and not structure:
            issues.append(PolicyIssue("Menu_InputAblation.py", getattr(node, "lineno", 1), f"{model_name} 启用 image 时必须显式声明 structure。"))
        if "image" not in active_inputs and structure:
            issues.append(PolicyIssue("Menu_InputAblation.py", getattr(node, "lineno", 1), f"{model_name} 未启用 image，不应声明图像结构标签。"))

    train_source = read_source("Train_core.py")
    forbidden_fallback = 'spec.structure or "high1+high2+high3+low"'
    if forbidden_fallback in train_source:
        issues.append(PolicyIssue("Train_core.py", line_of(train_source, forbidden_fallback), "Train_core 不得把空 structure 回退为 Full。"))
    for token in (
        "def resolve_required_structure_label",
        "def format_pyramid_setting_for_spec",
        '"Pyramid_Setting": format_pyramid_setting_for_spec(spec, active_inputs)',
        '"pyramid_label": format_pyramid_setting_for_spec(spec, active_inputs)',
    ):
        if token not in train_source:
            issues.append(PolicyIssue("Train_core.py", 1, f"Train_core 缺少 MFPC-HFNet 结构边界标记: {token}"))

    return issues


def check_menu_boundaries() -> list[PolicyIssue]:
    """
    检查菜单只声明训练清单。
    English: Check that menus only declare training lists.
    """

    issues: list[PolicyIssue] = []
    for file_name in MENU_FILES:
        source = read_source(file_name)
        for token in ("ExperimentMenu", "ModelSpec", "run_training_menu", "get_"):
            if token not in source:
                issues.append(PolicyIssue(file_name, 1, f"菜单缺少必要合同标记: {token}"))
        for token in MENU_FORBIDDEN_TOKENS:
            if token in source:
                issues.append(PolicyIssue(file_name, line_of(source, token), f"菜单不应承载训练主体或输出逻辑: {token}"))
    return issues


def check_metrics_boundary() -> list[PolicyIssue]:
    """
    检查上菜员层独立存在。
    English: Check that the metrics/output layer remains independent.
    """

    source = read_source("Metrics_core.py")
    issues: list[PolicyIssue] = []
    for token in ("calculate_detailed_metrics", "save_evaluation_results", "Times New Roman", "CSV"):
        if token not in source:
            issues.append(PolicyIssue("Metrics_core.py", 1, f"Metrics_core 缺少输出层标记: {token}"))
    train_support = read_source("Train_support.py")
    if "def calculate_detailed_metrics" in train_support or "def save_evaluation_results" in train_support:
        issues.append(PolicyIssue("Train_support.py", 1, "评价指标和评价报表不应留在 Train_support.py。"))
    return issues


def check_input_ablation_policy() -> list[PolicyIssue]:
    """
    检查输入端消融仍只包含 6 个待训练组合。
    English: Check that input-side ablation still contains only six trainable combinations.
    """

    source = read_source("Menu_InputAblation.py")
    issues: list[PolicyIssue] = []
    expected = [
        "InputAblation_NIROnly",
        "InputAblation_HyperOnly",
        "InputAblation_HyperNIR",
        "InputAblation_ImageOnly",
        "InputAblation_ImageNIR",
        "InputAblation_ImageHyper",
    ]
    for name in expected:
        if name not in source:
            issues.append(PolicyIssue("Menu_InputAblation.py", 1, f"输入端消融缺少模型: {name}"))
    if source.count('model_family="input_ablation"') != 6:
        issues.append(PolicyIssue("Menu_InputAblation.py", 1, "输入端消融应只声明 6 个待训练组合。"))
    if "InputAblation_Full" in source:
        issues.append(PolicyIssue("Menu_InputAblation.py", line_of(source, "InputAblation_Full"), "Full 不得进入输入端消融训练清单。"))
    return issues


def check_active_input_data_boundary() -> list[PolicyIssue]:
    """
    检查 active_inputs 是否贯穿训练库和正式训练 DataLoader。
    English: Check whether `active_inputs` is propagated through the data layer and formal training DataLoader.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    issues: list[PolicyIssue] = []
    data_source = read_source("Data_LoaderRuntimeAuto.py")
    for token in (
        "def normalize_active_inputs",
        "active_inputs=None",
        "self.active_inputs = normalize_active_inputs(active_inputs)",
        "required_cache_files_for_inputs",
        "check_sample_files(record['folder_path'], active_inputs=self.active_inputs)",
        "if \"image\" in self.active_inputs",
        "for key in self.active_inputs",
    ):
        if token not in data_source:
            issues.append(PolicyIssue("Data_LoaderRuntimeAuto.py", 1, f"Data 层缺少 active_inputs 根源裁剪标记: {token}"))

    registry_source = read_source("Data_DiskCacheRegistry.py")
    for token in (
        "active_inputs=None",
        "required_files=None",
        "required_files': list(required_files)",
        "for name in required_files",
    ):
        if token not in registry_source:
            issues.append(PolicyIssue("Data_DiskCacheRegistry.py", 1, f"缓存 registry 缺少按输入源检查必要文件能力: {token}"))

    train_source = read_source("Train_core.py")
    for token in (
        "active_inputs=normalize_spec_active_inputs(spec)",
        "active_inputs=model_active_inputs",
        "trim_inactive_inputs=True",
    ):
        if token not in train_source:
            issues.append(PolicyIssue("Train_core.py", 1, f"Train_core 缺少 active_inputs 训练链路标记: {token}"))
    forbidden = "train_loader = make_data_loader(dataset, split_indices[\"train\"], batch_size, True, loader_seed, device)"
    if forbidden in train_source:
        issues.append(PolicyIssue("Train_core.py", line_of(train_source, forbidden), "正式 train_loader 不得退回原始全模态 Subset。"))
    return issues


def run_all_checks() -> list[PolicyIssue]:
    """
    运行全部 V2 工程边界检查。
    English: Run all V2 engineering-boundary checks.
    """

    issues: list[PolicyIssue] = []
    issues.extend(check_required_files())
    issues.extend(check_no_forbidden_execution_split())
    issues.extend(check_train_core_boundary())
    issues.extend(check_optimizer_policy_boundary())
    issues.extend(check_train_main_config_boundary())
    issues.extend(check_mfpchf_structure_boundary())
    issues.extend(check_menu_boundaries())
    issues.extend(check_metrics_boundary())
    issues.extend(check_input_ablation_policy())
    issues.extend(check_active_input_data_boundary())
    return issues


def main() -> int:
    """
    命令行入口。
    English: Command-line entry point.
    """

    issues = run_all_checks()
    if issues:
        print("V2 training policy check FAILED:")
        for issue in issues:
            print(f"- {issue.file}:{issue.line}: {issue.message}")
        return 1
    print("V2 training policy check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
