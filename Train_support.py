# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
训练支撑工具库。
Training support utility library.

设计说明 / Design notes
English: Design notes.
----------------------
1. 本文件只保留训练过程支撑能力：随机种子、稳定折分、checkpoint 路径、断点文件读写和 train-loss patience 纯策略。
English: 1. filetraining: , , checkpoint path, file train-loss patience .
2. 评价指标和结果输出属于 `Metrics_core.py`，不再放在训练支撑工具里。
English: 2. metricresultOutput `Metrics_core.py`, training.
3. 训练入口、菜单合同和训练任务执行只从 `Train_core.py` 进入。
English: 3. training, menutraining `Train_core.py` .
4. 本文件不依赖菜单、模型或数据集的具体实现，避免支撑工具反向成为训练主体。
English: 4. filemenu, model, avoidtraining.

最近修改时间 / Last modified: 2026-05-29
English: Last modified: 2026-05-29.
作者 / Author: ljy
English: Author: ljy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    设置 Python / NumPy / PyTorch 随机种子。
    English: NumPy / PyTorch 随机种子.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def release_cuda_memory() -> None:
    """
    尝试释放 CUDA 缓存。
    English: CUDA cache.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_stable_split_id_from_item(item: dict) -> str:
    """
    从数据样本条目中提取稳定划分 ID。
    English: sample ID.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    for key in ("stable_split_id", "Stable_Split_ID", "sample_name", "Sample_Name", "folder_name", "Folder_Name"):
        value = item.get(key) if isinstance(item, dict) else None
        if value not in (None, ""):
            return str(value)
    return str(item)


def stable_hash_to_fold_id(stable_split_id: str, split_seed: int, num_folds: int) -> int:
    """
    将稳定样本 ID 映射到 fold 编号。
    English: sample ID fold .

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    text = f"{split_seed}|{stable_split_id}"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(num_folds)


def build_stable_fold_assignments(dataset: Any, split_seed: int, num_folds: int) -> list[list[int]]:
    """
    基于稳定样本 ID 构建交叉验证折分。
    English: sample ID build.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    records = getattr(dataset, "data_cache", dataset)
    folds: list[list[int]] = [[] for _ in range(int(num_folds))]
    for index, item in enumerate(records):
        stable_id = get_stable_split_id_from_item(item)
        fold_id = stable_hash_to_fold_id(stable_id, split_seed, num_folds)
        folds[fold_id].append(index)
    return folds


def get_split_indices_for_run(
    fold_assignments: Sequence[Sequence[int]],
    run_idx: int,
    validation_fold_offset: int,
) -> dict[str, Any]:
    """
    按运行折号生成 train / validation / test 索引。
    English: validation / test 索引.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    num_folds = len(fold_assignments)
    test_fold = int(run_idx) - 1
    val_fold = (test_fold + int(validation_fold_offset)) % num_folds
    train_indices: list[int] = []
    for fold_id, fold_indices in enumerate(fold_assignments):
        if fold_id not in {test_fold, val_fold}:
            train_indices.extend(int(item) for item in fold_indices)
    return {
        "train": train_indices,
        "val": [int(item) for item in fold_assignments[val_fold]],
        "test": [int(item) for item in fold_assignments[test_fold]],
        "val_fold": int(val_fold),
        "test_fold": int(test_fold),
    }


def get_dataset_stable_ids(dataset: Any) -> list[str]:
    """
    提取数据集稳定样本 ID 列表。
    English: sample ID list.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    return [get_stable_split_id_from_item(item) for item in getattr(dataset, "data_cache", dataset)]


def validate_dataset_alignment_for_shared_folds(dataset: Any, reference_stable_ids: Sequence[str], dataset_label: str) -> None:
    """
    校验当前数据集与 shared_folds 记录的样本顺序一致。
    English: validationcurrent shared_folds sample.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    current_stable_ids = get_dataset_stable_ids(dataset)
    if list(current_stable_ids) != list(reference_stable_ids):
        raise ValueError(f"Dataset alignment mismatch for {dataset_label}; shared fold records cannot be reused.")


def get_validation_history_path(run_dir: str) -> str:
    """
    返回 validation_history.csv 路径。
    English: return validation_history.csv path.
    """

    return os.path.join(run_dir, "validation_history.csv")


def get_training_progress_json_path(run_dir: str) -> str:
    """
    返回 training_progress.json 路径。
    English: return training_progress.json path.
    """

    return os.path.join(run_dir, "training_progress.json")


def get_resume_best_state_path(run_dir: str) -> str:
    """
    返回验证级续训 checkpoint 路径。
    English: return checkpoint path.
    """

    return os.path.join(run_dir, "resume_from_best_state.pth")


def get_best_model_path(run_dir: str) -> str:
    """
    返回 best_model.pth 路径。
    English: return best_model.pth path.
    """

    return os.path.join(run_dir, "best_model.pth")


def append_validation_history(run_dir: str, row: dict) -> None:
    """
    追加写入 validation_history.csv。
    English: write validation_history.csv.

    说明 / Notes:
    English: Notes:.
    - 续训旧 Fold 时优先沿用已有表头顺序，避免新旧日志列名顺序不一致导致 CSV 错列；
    English: - Fold , avoid CSV ;
    - 新增字段若旧表头不存在，会在旧表中忽略，完整信息仍保存在 training_progress.json。
    English: - field, , save training_progress.json.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    path = get_validation_history_path(run_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    fieldnames = list(row.keys())
    if file_exists:
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, None)
        if existing_header:
            fieldnames = list(existing_header)
    with open(path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_training_progress_json(run_dir: str, payload: dict) -> None:
    """
    保存训练进度 JSON。
    English: savetraining JSON.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    path = get_training_progress_json_path(run_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json_if_exists(path: str) -> Optional[dict]:
    """
    读取 JSON 文件，缺失时返回 None。
    English: read JSON file, missingreturn None.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    if not path or not os.path.isfile(path):
        return None
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_validation_log_row(epoch: int, avg_train_loss: float, lr: float, val_rmse: float, is_best: bool, **extra) -> dict:
    """
    构造验证历史记录行。
    English: This docstring documents the corresponding function behavior and engineering constraints.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    row = {
        "Epoch": int(epoch),
        "Train_Loss": float(avg_train_loss),
        "Learning_Rate": float(lr),
        "Val_RMSE": float(val_rmse),
        "Is_Best": bool(is_best),
    }
    row.update(extra)
    return row


def get_train_loss_patience_limits(base_patience_cycles: int, max_multiplier: float) -> dict:
    """
    计算训练损失临时 patience 上限。
    English: calculatetraining patience .

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    base = int(base_patience_cycles)
    return {
        "base_patience_cycles": base,
        "max_patience_cycles": int(round(base * float(max_multiplier))),
    }


def get_current_train_loss_patience_cycles(base_patience_cycles: int, current_bonus_cycles: int, max_multiplier: float) -> int:
    """
    计算当前 train-loss bonus 后的 patience。
    English: calculatecurrent train-loss bonus patience.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    limits = get_train_loss_patience_limits(base_patience_cycles, max_multiplier)
    return min(int(base_patience_cycles) + int(current_bonus_cycles), limits["max_patience_cycles"])


def update_train_loss_patience_after_epoch(
    epoch_train_loss: float,
    best_train_epoch_loss: float,
    current_bonus_cycles: int,
    base_patience_cycles: int,
    bonus_cycles: int,
    max_multiplier: float,
    enabled: bool = True,
) -> dict:
    """
    根据 epoch 训练损失刷新 train-loss patience bonus。
    English: epoch training train-loss patience bonus.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    if not enabled or not np.isfinite(epoch_train_loss):
        improved = False
    else:
        improved = float(epoch_train_loss) < float(best_train_epoch_loss)

    best_loss = float(epoch_train_loss) if improved else float(best_train_epoch_loss)
    bonus = int(current_bonus_cycles)
    if improved:
        max_bonus = get_train_loss_patience_limits(base_patience_cycles, max_multiplier)["max_patience_cycles"] - int(base_patience_cycles)
        bonus = min(max_bonus, bonus + int(bonus_cycles))
    return {
        "best_train_epoch_loss": best_loss,
        "train_loss_patience_bonus_cycles": bonus,
        "current_lr_patience_cycles": get_current_train_loss_patience_cycles(base_patience_cycles, bonus, max_multiplier),
        "train_loss_improved": improved,
    }


def reset_train_loss_patience_bonus(base_patience_cycles: int) -> dict:
    """
    在验证集 best 刷新或 LR decay 后清零 train-loss bonus。
    English: best LR decay train-loss bonus.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    return {
        "train_loss_patience_bonus_cycles": 0,
        "current_lr_patience_cycles": int(base_patience_cycles),
        "train_loss_patience_reset_reason": "validation_best_or_lr_decay",
    }


def export_best_train_epoch_loss(best_train_epoch_loss) -> Optional[float]:
    """
    导出可 JSON 序列化的 best train epoch loss。
    English: export JSON best train epoch loss.
    """

    if best_train_epoch_loss is None or not np.isfinite(best_train_epoch_loss):
        return None
    return float(best_train_epoch_loss)


__all__ = [
    "append_validation_history",
    "build_stable_fold_assignments",
    "build_validation_log_row",
    "export_best_train_epoch_loss",
    "get_best_model_path",
    "get_current_train_loss_patience_cycles",
    "get_dataset_stable_ids",
    "get_resume_best_state_path",
    "get_split_indices_for_run",
    "get_stable_split_id_from_item",
    "get_train_loss_patience_limits",
    "get_training_progress_json_path",
    "get_validation_history_path",
    "read_json_if_exists",
    "release_cuda_memory",
    "reset_train_loss_patience_bonus",
    "save_training_progress_json",
    "set_global_seed",
    "stable_hash_to_fold_id",
    "update_train_loss_patience_after_epoch",
    "validate_dataset_alignment_for_shared_folds",
]
