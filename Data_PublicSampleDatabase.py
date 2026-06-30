# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
公开版单文件样本数据库工具。
Public single-file sample database utilities.

逻辑说明 / Logic
English: Logic.
----------------
1. 公开版数据库以一个样本一个 `.npz` 文件保存，根目录包含清单文件和 `samples/` 子目录。
English: 1. The public database stores one `.npz` file per sample; the root contains a manifest and the `samples/` directory.
2. 每个样本文件包含 `sample_id`、多源输入数据、`labels` 和 `target_names`，可直接组成训练 Dataset 条目。
English: 2. Each sample file contains `sample_id`, multisource inputs, `labels`, and `target_names`, and can be converted directly into a training Dataset item.
3. 训练期使用 `sample_id` 作为稳定折分 ID、输出对齐 ID 和样本索引 ID。
English: 3. During training, `sample_id` is used as the stable split ID, output-alignment ID, and sample index ID.

最近修改时间 / Last modified: 2026-06-30
English: Last modified: 2026-06-30.
作者 / Author: ljy / GG
English: Author: ljy / GG.
维护记录 / Maintenance:
- 2026-06-30；作者：GG。说明文字改为当前公开数据库字段和训练期索引语义。
English: - 2026-06-30; Author: GG. Documentation now describes the current public database fields and training-time index semantics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


PUBLIC_DATABASE_FORMAT = "public_single_npz_v1"
PUBLIC_MANIFEST_NAME = "public_dataset_manifest.json"
PUBLIC_SAMPLES_DIR_NAME = "samples"


def normalize_public_database_root(data_root: str | os.PathLike) -> Path:
    """
    解析公开数据库根目录。
    English: Resolve the public database root directory.

    输入 / Inputs:
    English: Inputs:.
    - `data_root`: 可以指向数据库根目录，也可以直接指向 `samples` 子目录。
    English: - `data_root`: database root directory, `samples` directory.

    输出 / Outputs:
    English: Outputs:.
    - 返回包含 `public_dataset_manifest.json` 的数据库根目录。
    English: - return `public_dataset_manifest.json` database root directory.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    path = Path(data_root).expanduser()
    if (path / PUBLIC_MANIFEST_NAME).is_file():
        return path
    if path.name == PUBLIC_SAMPLES_DIR_NAME and (path.parent / PUBLIC_MANIFEST_NAME).is_file():
        return path.parent
    raise FileNotFoundError(
        f"未找到公开数据库清单 {PUBLIC_MANIFEST_NAME}。"
        f"请确认 DATA_DIR 或 DATASET_ROOT 指向公开数据库根目录或 samples 子目录: {path}"
    )


def is_public_sample_database(data_root: str | os.PathLike | None) -> bool:
    """
    判断路径是否为公开版单文件样本数据库。
    English: Check whether a path points to the public single-file sample database.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    if data_root in (None, ""):
        return False
    try:
        normalize_public_database_root(data_root)
        return True
    except FileNotFoundError:
        return False


def read_public_manifest(database_root: str | os.PathLike) -> dict[str, Any]:
    """
    读取公开数据库清单并执行最小格式校验。
    English: Read the public database manifest and perform minimal format validation.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    root = normalize_public_database_root(database_root)
    manifest_path = root / PUBLIC_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("database_format") != PUBLIC_DATABASE_FORMAT:
        raise ValueError(
            f"公开数据库格式不匹配: {manifest.get('database_format')!r}; "
            f"期望 {PUBLIC_DATABASE_FORMAT!r}。"
        )
    manifest["database_root"] = str(root)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def list_public_sample_files(database_root: str | os.PathLike) -> list[Path]:
    """
    返回公开数据库中的样本 `.npz` 文件列表。
    English: Return the list of `.npz` sample files in the public database.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    root = normalize_public_database_root(database_root)
    samples_dir = root / PUBLIC_SAMPLES_DIR_NAME
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"公开数据库缺少 samples 子目录: {samples_dir}")
    return sorted(samples_dir.glob("*.npz"))


def normalize_target_names(target_names: Iterable[Any]) -> list[str]:
    """
    规范化标签名列表。
    English: Normalize the target-name list.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    normalized = [str(item).strip().upper() for item in target_names if str(item).strip()]
    return normalized or ["SOC"]


def target_names_for_mode(target_mode: str) -> list[str]:
    """
    将训练目标模式转换为标签名列表。
    English: Convert the training target mode to a target-name list.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    mode = str(target_mode).strip().lower()
    if mode == "both":
        return ["SOC", "TN"]
    if mode == "tn":
        return ["TN"]
    return ["SOC"]


def labels_for_target_mode(labels: np.ndarray, available_targets: list[str], target_mode: str) -> np.ndarray:
    """
    从公开数据库样本标签中取出当前训练目标所需的标签向量。
    English: Extract the label vector required by the current target mode from a public-database sample.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    available = normalize_target_names(available_targets)
    requested = target_names_for_mode(target_mode)
    label_array = np.asarray(labels, dtype=np.float32).reshape(-1)
    values = []
    missing = []
    for target in requested:
        if target not in available:
            missing.append(target)
            continue
        values.append(float(label_array[available.index(target)]))
    if missing:
        raise ValueError(
            f"公开数据库不包含当前 TARGET_MODE={target_mode!r} 所需标签: {missing}。"
            f"可用标签: {available}"
        )
    if len(values) == 1:
        return np.asarray(values[0], dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def resize_image_tensor_if_needed(image_chw: np.ndarray, image_size: tuple[int, int] | None) -> torch.Tensor:
    """
    按菜单要求调整公开数据库中的图像尺寸。
    English: menupublic databaseimage.

    输入 / Inputs:
    English: Inputs:.
    - `image_chw`: `[C, H, W]` float32 图像。
    English: - `image_chw`: `[C, H, W]` float32 image.
    - `image_size`: 目标 `(H, W)`；None 表示保持原尺寸。
    English: - `image_size`: `(H, W)`; None .

    输出 / Outputs:
    - `torch.Tensor[C, H, W]`。

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    tensor = torch.from_numpy(np.asarray(image_chw, dtype=np.float32))
    if image_size is None:
        return tensor
    target_hw = (int(image_size[0]), int(image_size[1]))
    if tuple(tensor.shape[-2:]) == target_hw:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def load_public_sample_npz(
    sample_path: str | os.PathLike,
    active_inputs: tuple[str, ...],
    image_size: tuple[int, int] | None,
    nir_dim: int,
    target_mode: str,
) -> dict[str, Any]:
    """
    读取一个公开版 `.npz` 样本并转换成训练 Dataset 条目。
    English: readpublic release `.npz` sampletraining Dataset .

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    path = Path(sample_path)
    with np.load(path, allow_pickle=False) as data:
        sample_id = str(data["sample_id"].item())
        target_names = normalize_target_names(data["target_names"].tolist())
        labels = labels_for_target_mode(data["labels"], target_names, target_mode)
        item: dict[str, Any] = {
            "cache_mode": "public_npz",
            "sample_name": sample_id,
            "stable_split_id": sample_id,
            "core_id": sample_id,
            "folder_path": sample_id,
            "sample_file": str(path),
            "label": torch.tensor(labels, dtype=torch.float32),
        }
        if "hyper" in active_inputs:
            item["hyper"] = torch.from_numpy(np.asarray(data["hyper"], dtype=np.float32).copy())
        if "nir" in active_inputs:
            nir = np.asarray(data["nir"], dtype=np.float32).reshape(-1)
            if len(nir) > int(nir_dim):
                nir = nir[: int(nir_dim)]
            elif len(nir) < int(nir_dim):
                nir = np.concatenate([nir, np.zeros(int(nir_dim) - len(nir), dtype=np.float32)])
            item["nir"] = torch.from_numpy(nir.astype(np.float32, copy=False).copy())
        if "image" in active_inputs:
            item["image"] = resize_image_tensor_if_needed(data["image"], image_size)
    return item
