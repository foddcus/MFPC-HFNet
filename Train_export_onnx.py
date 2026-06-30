# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
V2 training-time ONNX export helpers.
V2 训练完成后 ONNX 导出辅助模块。
English: V2 training ONNX export.

Logic / 逻辑
English: Logic / Logic.
------------
1. Select one representative Fold after a model finishes all Fold outputs.
   在单个模型全部 Fold 完成后，选择一个代表 Fold。
2. Rebuild the model from the menu-owned ModelSpec and current Config.
   通过菜单持有的 ModelSpec 与当前 Config 重建模型，避免猜测结构。
3. Export only the active application inputs to ONNX.
   ONNX 只暴露当前模型实际启用的应用端输入。
4. Write a JSON manifest beside the ONNX file for deployment traceability.
   在 ONNX 旁写入 JSON 追溯文件，记录来源 Fold、权重、接口、数据库名称和模型复杂度。
   English: ONNX write JSON file, Fold, , , namemodel.

最近修改时间 / Last modified: 2026-06-17
English: Last modified: 2026-06-17.
作者 / Author: ljy / GG
English: Author: ljy / GG.
维护记录 / Maintenance:
English: Maintenance:.
- 2026-06-17；作者：GG。代表 Fold ONNX 导出优先读取该 Fold 的 train-only PCA 先验，
English: - 2026-06-17; Author: GG. Fold ONNX exportread Fold train-only PCA ,.
  缺失时再回退到全局 PCA_PRIORS_PATH。
  English: missingfall back PCA_PRIORS_PATH.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


VALID_INPUT_ORDER = ("hyper", "nir", "image")


def sanitize_filename_part(value: Any, fallback: str = "onnx_export") -> str:
    """
    清理 Windows 文件名非法字符。
    Sanitize one filename component for Windows paths.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    text = str(value or "").strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text).strip(" ._")
    return text or fallback


def normalize_target_mode_tag(config: Any) -> str:
    """
    返回用于文件名的目标变量标签，例如 SOC / TN / SOC_TN。
    Return the target tag used in ONNX filenames.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    target_names = get_target_names_for_export(config)
    return "_".join(str(name).upper() for name in target_names)


def get_target_names_for_export(config: Any) -> list[str]:
    """
    从 Config 中读取目标变量名称。
    Read target names from the active Config.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    if hasattr(config, "get_target_names"):
        return [str(item) for item in config.get_target_names()]
    mode = str(getattr(config, "TARGET_MODE", "soc")).lower()
    if mode == "both":
        return ["SOC", "TN"]
    if mode == "tn":
        return ["TN"]
    return ["SOC"]


def render_onnx_filename(template: str, spec: Any, config: Any, fold_name: str, metric_name: str) -> str:
    """
    按 Train_main 面板中的模板生成 ONNX 文件名。
    Render the ONNX filename from the Train_main template.

    支持变量 / Supported fields:
    English: Supported fields:.
    - {model_name}: 工程模型名，即 spec.name；
    English: - {model_name}: model, spec.name;
    - {display_name}: 展示名，自动清理非法字符；
    - {target_mode}: SOC / TN / SOC_TN；
    - {fold}: Fold01 等代表 Fold；
    English: - {fold}: Fold01 Fold;
    - {metric}: test_rmse 等选择指标。
    English: - {metric}: test_rmse selectmetric.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    template = str(template or "{model_name}_{target_mode}.onnx").strip() or "{model_name}_{target_mode}.onnx"
    values = {
        "model_name": sanitize_filename_part(getattr(spec, "name", ""), "model"),
        "display_name": sanitize_filename_part(getattr(spec, "display_name", ""), "model"),
        "target_mode": sanitize_filename_part(normalize_target_mode_tag(config), "SOC"),
        "fold": sanitize_filename_part(fold_name, "FoldXX"),
        "metric": sanitize_filename_part(metric_name, "metric"),
    }
    try:
        filename = template.format(**values)
    except KeyError as error:
        valid = ", ".join(sorted(values))
        raise KeyError(f"ONNX_EXPORT_NAME_TEMPLATE 包含未知变量 {error!s}；支持变量: {valid}") from error
    filename = sanitize_filename_part(filename, "model.onnx")
    if not filename.lower().endswith(".onnx"):
        filename += ".onnx"
    return filename


def select_representative_fold_by_test_rmse(model_dir: Path, target_names: Sequence[str]) -> dict[str, Any]:
    """
    按 Test RMSE 选择代表 Fold。
    Select the representative Fold by the lowest Test RMSE.

    设计说明：
    English: Design note:
    - 单目标时使用该目标的 Test RMSE；
    English: - Test RMSE;
    - 多目标时对目标 Test RMSE 取平均，避免只偏向某一目标；
    English: - Test RMSE , avoid;
    - 该选择策略会写入 export_info.json，因为用测试集选择部署权重有方法学含义。
    English: - selectwrite export_info.json, select.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    import pandas as pd

    target_name_set = {str(item) for item in target_names}
    candidates: list[dict[str, Any]] = []
    for fold_dir in sorted(Path(model_dir).glob("Fold*")):
        if not fold_dir.is_dir():
            continue
        metrics_path = fold_dir / "metrics_summary.csv"
        if not metrics_path.is_file():
            continue
        df = pd.read_csv(metrics_path, encoding="utf-8-sig")
        required_columns = {"Subset", "Target", "RMSE"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{metrics_path} 缺少代表 Fold 选择所需列: {sorted(missing)}")
        test_df = df[df["Subset"].astype(str).str.lower() == "test"].copy()
        test_df = test_df[test_df["Target"].astype(str).isin(target_name_set)]
        if test_df.empty:
            continue
        rmse_values = [float(value) for value in test_df["RMSE"].tolist() if math.isfinite(float(value))]
        if not rmse_values:
            continue
        candidates.append(
            {
                "fold_dir": fold_dir,
                "fold": fold_dir.name,
                "metric_name": "test_rmse",
                "metric_value": float(sum(rmse_values) / len(rmse_values)),
                "target_rmse": {
                    str(row["Target"]): float(row["RMSE"])
                    for _, row in test_df.iterrows()
                    if math.isfinite(float(row["RMSE"]))
                },
                "metrics_path": metrics_path,
            }
        )
    if not candidates:
        raise FileNotFoundError(f"未在 {model_dir} 下找到可用于 ONNX 代表 Fold 选择的 Test RMSE。")
    return min(candidates, key=lambda item: item["metric_value"])


def strip_thop_state_dict_keys(state_dict: Any) -> Any:
    """
    删除历史 THOP 可能注入的统计键。
    Remove THOP bookkeeping keys that should not be loaded as model weights.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    if not isinstance(state_dict, dict):
        return state_dict
    return {
        key: value
        for key, value in state_dict.items()
        if "total_ops" not in str(key) and "total_params" not in str(key)
    }


def count_model_parameters_for_export(model: Any) -> dict[str, Any]:
    """
    统计导出模型的参数量。
    Count model parameters for ONNX export metadata.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    维护说明：新增 fold_pca_prior 字段，记录代表 Fold 使用的 PCA 先验来源。
    English: : fold_pca_prior field, Fold PCA .
    """

    total = int(sum(param.numel() for param in model.parameters()))
    active = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    return {
        "params_total": total,
        "params_active_total": active,
        "params_registered_total": total,
        "params_m": total / 1_000_000.0,
        "params_active_m": active / 1_000_000.0,
        "params_registered_m": total / 1_000_000.0,
    }


def resolve_dataset_traceability_for_export(config: Any) -> dict[str, Any]:
    """
    解析 ONNX JSON 中的训练数据库追溯信息。
    Resolve training-database traceability fields for ONNX metadata.

    设计说明：
    English: Design note:
    - database_name 使用 DATASET_ROOT 的目录名，便于应用端和人工检查快速识别数据库版本；
    English: - database_name DATASET_ROOT directory, check;
    - 公开版导出不写入本机完整路径，避免 ONNX 旁路 JSON 泄露工作目录；
    English: - public releaseexportwritepath, avoid ONNX JSON directory;
    - dataset_root / data_dir 只写公开占位描述。
    English: data_dir 只写公开占位描述.

    最近修改时间：2026-06-16；作者：ljy。
    English: Last modified: 2026-06-16; Author: ljy.
    """

    dataset_root_text = str(getattr(config, "DATASET_ROOT", "") or "").strip()
    data_dir_text = str(getattr(config, "DATA_DIR", "") or "").strip()
    if not dataset_root_text and data_dir_text:
        data_path = Path(data_dir_text)
        dataset_root_text = str(data_path.parent) if data_path.name else data_dir_text
    database_name = Path(dataset_root_text).name if dataset_root_text else ""
    return {
        "database_name": database_name,
        "dataset_root": "<public_database_root>",
        "data_dir": "samples",
        "database_format": "public_single_npz_v1",
    }


def _json_float_or_none(value: Any) -> float | None:
    """
    将 CSV/JSON 中的数值字段安全转成 float。
    Safely convert CSV/JSON metric values to float.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    维护说明：重建模型前优先读取代表 Fold 的 train-only PCA 先验，保证 ONNX 与训练 Fold 统计先验一致。
    English: : modelread Fold train-only PCA , ensure ONNX training Fold .
    """

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def read_representative_fold_flops_g(metrics_path: Path) -> tuple[float, str]:
    """
    从代表 Fold 的 metrics_summary.csv 读取 FLOPs_G。
    Read FLOPs_G from the representative Fold metrics_summary.csv.

    设计说明：
    English: Design note:
    - 本轮按用户确认沿用现有训练输出中的 FLOPs_G，不新增真实 FLOPs 计算；
    English: - trainingOutput FLOPs_G, FLOPs calculate;
    - 当前训练代码多数记录为 0.0，占位值仍原样进入 JSON；
    English: - currenttraining 0.0, JSON;
    - 读不到字段或文件时写 0.0，并在 complexity_source 中记录缺失来源。
    English: - fieldfile 0.0, complexity_source missing.

    最近修改时间：2026-06-07；作者：ljy。
    English: Last modified: 2026-06-07; Author: ljy.
    """

    metrics_path = Path(metrics_path)
    if not metrics_path.is_file():
        return 0.0, "missing_metrics_summary_csv_default_0"
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "FLOPs_G" not in reader.fieldnames:
            return 0.0, "missing_FLOPs_G_column_default_0"
        for row in reader:
            value = _json_float_or_none(row.get("FLOPs_G"))
            if value is not None:
                return float(value), str(metrics_path)
    return 0.0, "empty_FLOPs_G_default_0"


def calculate_model_flops_g_for_export(model: Any, spec: Any, config: Any, active_inputs: Sequence[str], target_names: Sequence[str]) -> tuple[float, str]:
    """
    使用 THOP 统计当前 ONNX 输入接口下的模型 FLOPs。
    Calculate model FLOPs with THOP under the active ONNX input interface.

    设计说明：
    English: Design note:
    - 统计输入与 ONNX tracing 输入一致，保证 JSON 中 FLOPs 对应部署接口；
    English: - Input ONNX tracing Input, ensure JSON FLOPs ;
    - THOP 返回的 ops 口径按常见论文表述折算为 FLOPs_G = 2 * MACs / 1e9；
    English: 1e9；.
    - 若 THOP 不可用或模型存在不支持算子，则回退到既有 metrics_summary.csv 读取逻辑。
    English: - THOP model, fall back metrics_summary.csv readLogic.

    最近修改时间：2026-06-07；作者：ljy。
    English: Last modified: 2026-06-07; Author: ljy.
    """

    import torch
    from thop import profile

    export_device = torch.device("cpu")
    model.to(export_device)
    model.eval()
    dummy_inputs = build_dummy_inputs(active_inputs, spec, config, export_device)
    export_model = ONNXActiveInputWrapper(model, active_inputs, target_names).to(export_device)
    export_model.eval()
    macs, _ = profile(export_model, inputs=dummy_inputs, verbose=False)
    return float(macs) * 2.0 / 1_000_000_000.0, "thop_profile_active_onnx_inputs_flops_2x_macs"


def load_model_weights_for_export(model: Any, pth_path: Path) -> None:
    """
    从 best_model.pth 读取权重并严格加载到重建模型。
    Load weights from best_model.pth into the rebuilt model with strict matching.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    import torch

    loaded = torch.load(str(pth_path), map_location="cpu")
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        loaded = loaded["model_state_dict"]
    clean_state_dict = strip_thop_state_dict_keys(loaded)
    model.load_state_dict(clean_state_dict, strict=True)


def resolve_fold_pca_prior_for_export(fold_dir: Path, spec: Any, config: Any) -> dict[str, Any]:
    """
    读取代表 Fold 的 train-only PCA 先验路径，供 ONNX 重建模型使用。
    English: read Fold train-only PCA path, ONNX model.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    fold_dir = Path(fold_dir)
    run_info_path = fold_dir / "run_info.json"
    if run_info_path.is_file():
        try:
            run_info = json.loads(run_info_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            run_info = {}
        fold_prior = dict(run_info.get("fold_pca_prior") or {})
        prior_path_text = str(fold_prior.get("priors_path") or run_info.get("pca_priors_path") or "").strip()
        if prior_path_text:
            raw_path = Path(prior_path_text)
            candidates = [raw_path]
            if not raw_path.is_absolute():
                candidates.append(fold_dir / raw_path)
            candidates.append(fold_dir / raw_path.name)
            for candidate in candidates:
                if candidate.is_file():
                    fold_prior.update({
                        "enabled": True,
                        "policy": fold_prior.get("policy", "fold_train_only_pca_priors"),
                        "priors_path": str(candidate),
                        "run_info_path": str(run_info_path),
                        "export_resolution": "representative_fold_run_info",
                    })
                    return fold_prior

    fallback_path = str(getattr(spec, "priors_path", "") or getattr(config, "PCA_PRIORS_PATH", "") or "").strip()
    return {
        "enabled": False,
        "policy": "global_config_pca_priors_fallback",
        "priors_path": fallback_path,
        "run_info_path": str(run_info_path),
        "export_resolution": "missing_or_unusable_fold_train_only_prior",
    }


def resolve_export_active_inputs(model: Any, spec: Any) -> tuple[str, ...]:
    """
    解析 ONNX 应暴露的实际输入源。
    Resolve the active application inputs exposed by the ONNX model.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    from Train_core import resolve_model_active_inputs

    active_inputs = resolve_model_active_inputs(model, spec)
    normalized = tuple(str(item).strip().lower() for item in active_inputs if str(item).strip())
    invalid = [item for item in normalized if item not in VALID_INPUT_ORDER]
    if invalid:
        raise ValueError(f"ONNX 导出发现不支持的 active_inputs: {invalid}")
    active_set = set(normalized)
    return tuple(name for name in VALID_INPUT_ORDER if name in active_set)


def build_dummy_inputs(active_inputs: Sequence[str], spec: Any, config: Any, device: Any) -> tuple[Any, ...]:
    """
    按实际输入接口构造 ONNX tracing dummy inputs。
    Build dummy inputs according to the active ONNX input interface.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    import torch

    image_h, image_w = tuple(getattr(spec, "image_size", (1024, 1024)))
    shape_by_input = {
        "hyper": (1, int(getattr(config, "HYPER_DIM", 681))),
        "nir": (1, int(getattr(config, "NIR_DIM", 5))),
        "image": (
            1,
            int(getattr(config, "IMAGE_CHANNELS", 8)),
            int(image_h),
            int(image_w),
        ),
    }
    return tuple(torch.randn(*shape_by_input[name], dtype=torch.float32, device=device) for name in active_inputs)


class ONNXActiveInputWrapper:
    """
    将可变 ONNX 输入映射回模型固定 forward(hyper, nir, image) 接口。
    Map variable ONNX inputs back to the fixed model forward(hyper, nir, image).

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    def __new__(cls, model: Any, active_inputs: Sequence[str], target_names: Sequence[str]):
        import torch.nn as nn

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.active_inputs = tuple(active_inputs)
                self.target_names = [str(name).lower() for name in target_names]

            def forward(self, *actual_inputs):
                tensors = {"hyper": None, "nir": None, "image": None}
                for name, tensor in zip(self.active_inputs, actual_inputs):
                    tensors[name] = tensor
                output = self.model(tensors["hyper"], tensors["nir"], tensors["image"])
                if len(self.target_names) == 1:
                    return output
                return tuple(output[:, index] for index in range(len(self.target_names)))

        return _Wrapper()


def export_model_to_onnx(
    *,
    model: Any,
    spec: Any,
    config: Any,
    pth_path: Path,
    onnx_path: Path,
    active_inputs: Sequence[str],
    target_names: Sequence[str],
    opset_version: int,
    dynamic_batch: bool,
) -> None:
    """
    执行 torch.onnx.export。
    Run torch.onnx.export with the project export policy.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    import torch

    export_device = torch.device("cpu")
    model.to(export_device)
    load_model_weights_for_export(model, pth_path)
    model.eval()

    dummy_inputs = build_dummy_inputs(active_inputs, spec, config, export_device)
    export_model = ONNXActiveInputWrapper(model, active_inputs, target_names).to(export_device)
    export_model.eval()

    output_names = [str(name).lower() for name in target_names]
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {str(name): {0: "B"} for name in active_inputs}
        for output_name in output_names:
            dynamic_axes[output_name] = {0: "B"}

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            export_model,
            dummy_inputs,
            str(onnx_path),
            input_names=list(active_inputs),
            output_names=output_names,
            dynamic_axes=dynamic_axes if dynamic_axes else None,
            opset_version=int(opset_version),
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            export_model,
            dummy_inputs,
            str(onnx_path),
            input_names=list(active_inputs),
            output_names=output_names,
            dynamic_axes=dynamic_axes if dynamic_axes else None,
            opset_version=int(opset_version),
            do_constant_folding=True,
        )


def write_export_info(
    *,
    info_path: Path,
    spec: Any,
    config: Any,
    selected_fold: dict[str, Any],
    pth_path: Path,
    onnx_path: Path,
    active_inputs: Sequence[str],
    target_names: Sequence[str],
    complexity_info: dict[str, Any],
    opset_version: int,
    dynamic_batch: bool,
    filename_template: str,
    fold_pca_prior: dict[str, Any] | None = None,
) -> None:
    """
    写入 ONNX 导出追溯信息。
    Write the ONNX export traceability manifest.

    最近修改时间：2026-06-07；作者：ljy。
    English: Last modified: 2026-06-07; Author: ljy.
    """

    dataset_info = resolve_dataset_traceability_for_export(config)
    fold_pca_prior = dict(fold_pca_prior or {"enabled": False, "policy": "not_recorded"})
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "exporter": "Train_export_onnx.py",
        "author": "ljy",
        "model_name": getattr(spec, "name", ""),
        "display_name": getattr(spec, "display_name", ""),
        "model_family": getattr(spec, "model_family", ""),
        **dataset_info,
        "target_mode": str(getattr(config, "TARGET_MODE", "soc")),
        "target_names": list(target_names),
        "selected_fold": selected_fold["fold"],
        "selection_metric": selected_fold["metric_name"],
        "selection_metric_value": selected_fold["metric_value"],
        "target_rmse": selected_fold["target_rmse"],
        "metrics_path": str(selected_fold["metrics_path"]),
        "best_model_path": str(pth_path),
        "onnx_path": str(onnx_path),
        "onnx_name_template": str(filename_template),
        "active_inputs": list(active_inputs),
        "input_names": list(active_inputs),
        "output_names": [str(name).lower() for name in target_names],
        "image_size": list(getattr(spec, "image_size", ())),
        "pca_prior_policy": str(fold_pca_prior.get("policy", "")),
        "pca_priors_path": str(fold_pca_prior.get("priors_path", "")),
        "fold_pca_prior": fold_pca_prior,
        **complexity_info,
        "opset_version": int(opset_version),
        "dynamic_batch": bool(dynamic_batch),
        "note": "Representative ONNX is selected by Test RMSE as requested; this is a deployment artifact selection policy.",
    }
    info_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_representative_onnx_for_model(spec: Any, config: Any, model_dir: Path) -> dict[str, Any] | None:
    """
    为单个模型目录导出代表 ONNX。
    Export one representative ONNX file for a completed model directory.

    最近修改时间：2026-06-07；作者：ljy。
    English: Last modified: 2026-06-07; Author: ljy.
    """

    if not bool(getattr(config, "EXPORT_ONNX_AFTER_TRAINING", False)):
        return None

    selection_policy = str(getattr(config, "ONNX_EXPORT_SELECT_FOLD_BY", "test_rmse")).strip().lower()
    if selection_policy != "test_rmse":
        raise ValueError(f"当前仅支持 ONNX_EXPORT_SELECT_FOLD_BY='test_rmse'，收到: {selection_policy!r}")

    from Train_core import build_model_from_spec, clone_model_spec

    target_names = get_target_names_for_export(config)
    selected_fold = select_representative_fold_by_test_rmse(Path(model_dir), target_names)
    fold_dir = Path(selected_fold["fold_dir"])
    pth_path = fold_dir / "best_model.pth"
    if not pth_path.is_file():
        raise FileNotFoundError(
            f"代表 Fold {fold_dir.name} 缺少 best_model.pth: {pth_path}。"
            "请确认 KEEP_COMPLETED_FOLD_BEST_MODEL=True，或重新保留该 Fold 权重后再导出。"
        )

    fold_pca_prior = resolve_fold_pca_prior_for_export(fold_dir, spec, config)
    export_spec = spec
    if str(fold_pca_prior.get("priors_path", "")).strip():
        export_spec = clone_model_spec(spec, priors_path=str(fold_pca_prior["priors_path"]))

    model = build_model_from_spec(export_spec, config)
    active_inputs = resolve_export_active_inputs(model, export_spec)
    complexity_info = count_model_parameters_for_export(model)
    filename_template = str(getattr(config, "ONNX_EXPORT_NAME_TEMPLATE", "{model_name}_{target_mode}.onnx"))
    onnx_filename = render_onnx_filename(
        filename_template,
        spec=export_spec,
        config=config,
        fold_name=str(selected_fold["fold"]),
        metric_name=str(selected_fold["metric_name"]),
    )
    output_dir = Path(model_dir) / "ONNX"
    onnx_path = output_dir / onnx_filename
    info_path = output_dir / f"{onnx_filename}.export_info.json"

    export_model_to_onnx(
        model=model,
        spec=export_spec,
        config=config,
        pth_path=pth_path,
        onnx_path=onnx_path,
        active_inputs=active_inputs,
        target_names=target_names,
        opset_version=int(getattr(config, "ONNX_EXPORT_OPSET", 18)),
        dynamic_batch=bool(getattr(config, "ONNX_EXPORT_DYNAMIC_BATCH", True)),
    )
    try:
        flops_g, flops_source = calculate_model_flops_g_for_export(model, export_spec, config, active_inputs, target_names)
        flops_policy = "thop_profile_flops_2x_macs"
    except Exception as error:
        flops_g, flops_source = read_representative_fold_flops_g(Path(selected_fold["metrics_path"]))
        flops_source = f"{flops_source}; thop_failed={type(error).__name__}: {error}"
        flops_policy = "fallback_existing_training_output_after_thop_failure"
    complexity_info["flops_g"] = float(flops_g)
    complexity_info["complexity_source"] = {
        "params": "rebuilt_model_parameter_count_for_onnx_export",
        "flops_g": flops_source,
        "flops_policy": flops_policy,
    }
    write_export_info(
        info_path=info_path,
        spec=export_spec,
        config=config,
        selected_fold=selected_fold,
        pth_path=pth_path,
        onnx_path=onnx_path,
        active_inputs=active_inputs,
        target_names=target_names,
        complexity_info=complexity_info,
        opset_version=int(getattr(config, "ONNX_EXPORT_OPSET", 18)),
        dynamic_batch=bool(getattr(config, "ONNX_EXPORT_DYNAMIC_BATCH", True)),
        filename_template=filename_template,
        fold_pca_prior=fold_pca_prior,
    )
    return {
        "onnx_path": str(onnx_path),
        "export_info_path": str(info_path),
        "selected_fold": str(selected_fold["fold"]),
        "selection_metric": str(selected_fold["metric_name"]),
        "selection_metric_value": float(selected_fold["metric_value"]),
        "active_inputs": list(active_inputs),
        "output_names": [str(name).lower() for name in target_names],
    }


__all__ = [
    "export_representative_onnx_for_model",
    "render_onnx_filename",
    "select_representative_fold_by_test_rmse",
]
