# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
训练输出与评价指标工具。
Metrics and result-output utilities.

逻辑说明 / Logic
English: Logic.
----------------
1. 本文件承接“上菜员”层职责：评价指标计算、回归散点图、评价报表和通用 CSV 汇总输出。
English: 1. file“”: metriccalculate, , general CSV Output.
2. `Train_*.py` 训练引擎只负责训练流程调度，不在工具层重复定义基础指标算法。
English: 2. `Train_*.py` training enginetraining, metric.
3. 画图统一使用 Times New Roman，图片和表格均写入调用方传入的运行目录子文件夹。
English: 3. Times New Roman, writedirectoryfile.
4. V2 为独立源码，不从原 `Python script` 目录导入任何旧代码。
English: 4. V2 , `Python script` directory.
5. 训练完成后的聚合表由本文件从 Fold 输出重建，样本名后处理只调用 `Data_` 前缀模块。
English: 5. trainingfile Fold Output, sample `Data_` .

最近修改时间 / Last modified: 2026-05-29
English: Last modified: 2026-05-29.
作者 / Author: ljy
English: Author: ljy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


def calculate_detailed_metrics(y_true, y_pred) -> dict:
    """
    计算回归任务常用评价指标。
    Calculate common regression metrics.

    输入 / Inputs:
    English: Inputs:.
    - `y_true`: 真实值数组；
    English: - `y_true`: ;
    - `y_pred`: 预测值数组。
    English: - `y_pred`: .

    输出 / Outputs:
    English: Outputs:.
    - 返回包含 R2、RMSE、MSE、MAE、MAPE 和 Explained Variance 的字典。
    English: - return R2, RMSE, MSE, MAE, MAPE Explained Variance dictionary.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if np.any(mask) else float("nan")
    return {
        "R2_Score": r2,
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "MAPE (%)": mape,
        "Explained_Variance": ev,
    }


def save_evaluation_results(y_true, y_pred, subset_name, save_dir, target_name="SOC") -> dict:
    """
    保存单个数据子集的评价图与 Excel 报表。
    Save one subset's regression plot and Excel report.

    物理意义 / Meaning:
    English: Meaning:.
    - 散点图用于观察预测值与真实值是否接近 1:1 线；
    English: - 1:1 ;
    - Excel 同时保存指标与散点坐标，便于后续重画或导入其他软件。
    English: - Excel savemetric, .

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    metrics = calculate_detailed_metrics(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="w", s=80, color="royalblue", label="Data Points")
    vmin = min(np.min(y_true), np.min(y_pred))
    vmax = max(np.max(y_true), np.max(y_pred))
    plt.plot([vmin, vmax], [vmin, vmax], "r--", lw=2.5, label="1:1 Ideal Line")
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), "g-.", lw=1.5, alpha=0.8, label=f"Fit: y={z[0]:.2f}x+{z[1]:.2f}")
    plt.xlabel(f"True {target_name} Value", fontsize=12)
    plt.ylabel(f"Predicted {target_name} Value", fontsize=12)
    plt.title(
        f"{subset_name} Set Regression Analysis ({target_name})\n"
        f"$R^2$={metrics['R2_Score']:.4f}, RMSE={metrics['RMSE']:.4f}",
        fontsize=14,
    )
    plt.legend(loc="upper left")
    plot_filename = f"{subset_name.lower()}_{target_name.lower()}_regression_plot.png"
    plt.savefig(os.path.join(save_dir, plot_filename), dpi=300, bbox_inches="tight")
    plt.close()

    excel_filename = f"{subset_name.lower()}_{target_name.lower()}_evaluation_report.xlsx"
    excel_path = os.path.join(save_dir, excel_filename)
    df_metrics = pd.DataFrame([metrics]).T
    df_metrics.columns = ["Value"]
    df_data = pd.DataFrame({
        "Sample_Index": range(len(y_true)),
        f"True_{target_name}": y_true,
        f"Predicted_{target_name}": y_pred,
        "Absolute_Error": np.abs(y_true - y_pred),
    })
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="Metrics")
        df_data.to_excel(writer, sheet_name="Scatter_Coordinates", index=False)
    print(f"    [Excel] {target_name} 评价报告已保存: {excel_filename}")
    print(f"    [Plot]  {target_name} 回归散点图已保存: {plot_filename}")
    return metrics


def round_float_columns(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """
    按统一小数位数复制并格式化 DataFrame 中的浮点列。
    English: DataFrame .

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    rounded = df.copy()
    float_cols = rounded.select_dtypes(include=["float", "float32", "float64"]).columns
    if len(float_cols) > 0:
        rounded[float_cols] = rounded[float_cols].round(decimals)
    return rounded


def save_csv_rounded(df: pd.DataFrame, path: str, decimals: int) -> None:
    """
    保存带统一浮点小数位的 CSV 汇总表。
    English: save CSV .

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    round_float_columns(df, decimals).to_csv(path, index=False, encoding="utf-8-sig")


def format_summary_value(values: Iterable[float], decimals: int = 2) -> str:
    """
    把一组折级指标格式化为 mean ± std 文本。
    English: metric mean ± std .

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    arr = np.array(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return ""
    return f"{arr.mean():.{decimals}f} ± {arr.std(ddof=1 if arr.size > 1 else 0):.{decimals}f}"


VALUE_PM_STD_COLUMNS = [
    "Best_Val_RMSE",
    "Params_M",
    "Params_Active_M",
    "Params_Registered_M",
    "FLOPs_G",
    "R2_Score",
    "RMSE",
    "MSE",
    "MAE",
    "MAPE (%)",
    "Explained_Variance",
]


def normalize_metric_summary_columns_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    将当前 V2 细粒度指标列补齐为历史汇总表使用的展示列名。
    Normalize current V2 metric columns to historical summary-display names.

    说明 / Notes:
    English: Notes:.
    - 当前 Fold 级表保留 `params_m` 等小写字段；
    English: - current Fold `params_m` field;
    - 聚合表为了兼容旧汇总格式，额外映射为 `Params_M` 等展示列；
    English: - compatible, `Params_M` ;
    - 本函数只复制/补齐列，不删除 Fold 级原始字段。
    English: - /, Fold field.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    normalized = df.copy()
    aliases = {
        "params_m": "Params_M",
        "params_active_m": "Params_Active_M",
        "params_registered_m": "Params_Registered_M",
    }
    for source, target in aliases.items():
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    return normalized


def build_value_pm_std_dataframe(metrics_df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """
    从 Fold 级指标表生成测试集 value±std 汇总表。
    Build the final Test-set value±std summary from fold-level metrics.

    物理意义 / Meaning:
    English: Meaning:.
    - 每个 Target 的一行表示所有已完成 Fold 的测试集均值和标准差；
    English: - Target Fold ;
    - 统计对象只取 `Subset == "Test"`，与旧版 `final_test_metrics_value_pm_std.csv` 口径一致。
    English: - `Subset == "Test"`, `final_test_metrics_value_pm_std.csv` .

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    if metrics_df.empty or "Subset" not in metrics_df.columns or "Target" not in metrics_df.columns:
        return pd.DataFrame()

    export_df = normalize_metric_summary_columns_for_export(metrics_df)
    test_df = export_df[export_df["Subset"] == "Test"].copy()
    if test_df.empty:
        return pd.DataFrame()

    numeric_cols = [col for col in VALUE_PM_STD_COLUMNS if col in test_df.columns]
    rows = []
    for target_name, group in test_df.groupby("Target", sort=False):
        row = {"Target": target_name}
        for col in numeric_cols:
            row[col] = format_summary_value(group[col], decimals=decimals)
        rows.append(row)
    return pd.DataFrame(rows)


def infer_prediction_target_from_filename(path: Path, target_names: list[str]) -> str:
    """
    从预测 CSV 文件名推断单目标旧格式对应的 Target。
    Infer the target name for compact historical prediction CSV files.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    match = re.search(r"_([A-Za-z0-9]+)\.csv$", path.name)
    if match:
        suffix = match.group(1).upper()
        for target_name in target_names:
            if suffix == str(target_name).upper():
                return str(target_name)
    return str(target_names[0]) if target_names else "SOC"


def read_prediction_rows_for_aggregation(path: Path, target_names: list[str]) -> list[dict]:
    """
    读取单个 Fold 的预测 CSV，并统一转换为聚合用长表记录。
    Read one fold prediction CSV and convert it into aggregate rows.

    支持格式 / Supported formats:
    English: Supported formats:.
    - 当前 V2 详细格式：`Sample_Name, True_SOC, Predicted_SOC, ...`；
    English: - current V2 : `Sample_Name, True_SOC, Predicted_SOC, ...`;
    - 旧版简洁格式：`SampleName, True, Pred`。
    English: - : `SampleName, True, Pred`.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    df = pd.read_csv(path, encoding="utf-8-sig")
    rows: list[dict] = []
    sample_column = "Sample_Name" if "Sample_Name" in df.columns else "SampleName"

    if {"SampleName", "True", "Pred"}.issubset(df.columns):
        target_name = infer_prediction_target_from_filename(path, target_names)
        for _, row in df.iterrows():
            rows.append({
                "SampleName": row.get("SampleName", ""),
                "Target": target_name,
                "True": row.get("True", np.nan),
                "Pred": row.get("Pred", np.nan),
            })
        return rows

    for target_name in target_names:
        true_col = f"True_{target_name}"
        pred_col = f"Predicted_{target_name}"
        if true_col not in df.columns or pred_col not in df.columns:
            continue
        for _, row in df.iterrows():
            rows.append({
                "SampleName": row.get(sample_column, ""),
                "Target": target_name,
                "True": row.get(true_col, np.nan),
                "Pred": row.get(pred_col, np.nan),
            })
    return rows


def save_model_aggregate_outputs(
    model_dir: str | os.PathLike,
    target_names: Iterable[str],
    decimals: int,
    sample_name_postprocess_enabled: bool = True,
) -> dict[str, str]:
    """
    从模型目录下已完成 Fold 重建聚合输出文件。
    Rebuild aggregate output files from completed Fold directories.

    输出 / Outputs:
    English: Outputs:.
    - `final_test_metrics_value_pm_std.csv`: 测试集指标 value±std 汇总；
    English: - `final_test_metrics_value_pm_std.csv`: metric value±std ;
    - `test_predictions_all_folds.csv`: 所有已完成 Fold 的测试集预测汇总。
    English: - `test_predictions_all_folds.csv`: Fold .

    设计选择 / Design choices:
    English: Design choices:.
    - 聚合阶段只读取磁盘上已经存在的 `metrics_summary.csv` 和 `test_predictions_fold_*.csv`；
    English: - read `metrics_summary.csv` `test_predictions_fold_*.csv`;
    - 若某个 Fold 未完成，则不会阻塞其他已完成 Fold 的聚合；
    English: - Fold , Fold ;
    - 公开版数据库已经在数据构建阶段完成匿名命名，聚合阶段不再执行旧样本名后处理。
    English: - public releasebuild, sample.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    del sample_name_postprocess_enabled  # 公开版训练输出直接使用 Dataset 中的匿名样本名。 / EN: public releasetrain directlyuse Dataset in sample.

    model_root = Path(model_dir)
    target_name_list = [str(item) for item in target_names]
    metrics_frames = []
    prediction_rows: list[dict] = []

    for fold_dir in sorted(model_root.glob("Fold*")):
        if not fold_dir.is_dir():
            continue

        metrics_path = fold_dir / "metrics_summary.csv"
        if metrics_path.is_file():
            metrics_frames.append(pd.read_csv(metrics_path, encoding="utf-8-sig"))

        for prediction_path in sorted(fold_dir.glob("test_predictions_fold_*.csv")):
            prediction_rows.extend(read_prediction_rows_for_aggregation(prediction_path, target_name_list))

    written_paths: dict[str, str] = {}
    if metrics_frames:
        metrics_df = pd.concat(metrics_frames, axis=0, ignore_index=True)
        final_test_df = build_value_pm_std_dataframe(metrics_df, decimals=decimals)
        if not final_test_df.empty:
            final_test_path = model_root / "final_test_metrics_value_pm_std.csv"
            save_csv_rounded(final_test_df, str(final_test_path), decimals)
            written_paths["final_test_metrics_value_pm_std"] = str(final_test_path)

    if prediction_rows:
        prediction_df = pd.DataFrame(prediction_rows)
        if len(target_name_list) == 1 and "Target" in prediction_df.columns:
            prediction_df = prediction_df.drop(columns=["Target"])
        prediction_df = prediction_df[["SampleName", *([col for col in ["Target"] if col in prediction_df.columns]), "True", "Pred"]]
        prediction_path = model_root / "test_predictions_all_folds.csv"
        save_csv_rounded(prediction_df, str(prediction_path), decimals)
        written_paths["test_predictions_all_folds"] = str(prediction_path)

    return written_paths


__all__ = [
    "calculate_detailed_metrics",
    "build_value_pm_std_dataframe",
    "normalize_metric_summary_columns_for_export",
    "read_prediction_rows_for_aggregation",
    "save_evaluation_results",
    "save_model_aggregate_outputs",
    "round_float_columns",
    "save_csv_rounded",
    "format_summary_value",
]
