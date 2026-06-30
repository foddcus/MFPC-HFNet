# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
公开版单文件样本数据库重构脚本。
Public single-file sample database rebuild script.

逻辑说明 / Logic
English: Logic.
----------------
1. 将旧版“批次文件夹 + 同级 blank + 外部标签表”的数据库重构为公开版单文件样本库。
English: 1. “file + blank + label”public releasefilesample.
2. 每个有效样本写成一个 `.npz` 文件，文件名和样本名均为 8 位数字/字母随机组合。
English: 2. sample `.npz` file, filesample 8 /.
3. 输出样本只包含 `image`、`hyper`、`nir`、`labels`、`target_names` 和匿名 `sample_id`，不保留原始样本名、原始路径或匿名映射表。
English: 3. Outputsample `image`, `hyper`, `nir`, `labels`, `target_names` `sample_id`, sample, path.
4. 标签只读取用户指定的 Excel 标签表；当前公开版数据库只写入 SOC 标签。
English: 4. labelread Excel label; currentpublic releasewrite SOC label.
5. 若存在旧训练流程已校正好的磁盘缓存，可优先从缓存重打包，避免重复读取 TIFF 和重复 blank 校正；否则退回原始库扫描和校正。
English: 5. trainingcache, cache, avoidread TIFF blank ; .

最近修改时间 / Last modified: 2026-06-16
English: Last modified: 2026-06-16.
作者 / Author: ljy
English: Author: ljy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import secrets
import shutil
import string
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from Data_LoaderRuntimeAuto import (
    WAVELENGTHS,
    check_sample_files,
    extract_core_id,
    read_hyper_csv,
    read_images,
    read_nir_csv,
)
from Data_PublicSampleDatabase import (
    PUBLIC_DATABASE_FORMAT,
    PUBLIC_MANIFEST_NAME,
    PUBLIC_SAMPLES_DIR_NAME,
)


DEFAULT_OUTPUT_ROOT = Path.home() / "Desktop" / "PublicSoilSampleDatabase"
DEFAULT_CACHE_SUBDIR = Path("SOC_SoilData") / "img_1024x1024__nir_5__soc__ch_8"
ALPHANUMERIC = string.ascii_uppercase + string.digits


def normalize_label_sample_name(sample_name: Any) -> str:
    """
    将旧采集样本名转换为标签表中的样本名口径。
    English: samplelabelsample.

    说明 / Notes:
    English: Notes:.
    - 该逻辑仅用于一次性数据库重构阶段；
    English: - Logic;
    - 公开版训练输出不再执行旧样本名后处理。
    English: - public releasetrainingOutputsample.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    text = str(sample_name).strip()
    if not text or text.lower() == "nan":
        return ""
    parts = [part.strip() for part in text.split("-") if part.strip()]
    if not parts:
        return text

    def is_ax_token(token: str) -> bool:
        return bool(re.fullmatch(r"A\d+", str(token).strip().upper()))

    def is_h_family_token(token: str) -> bool:
        token_upper = str(token).strip().upper()
        return token_upper == "H" or bool(re.fullmatch(r"H\d+", token_upper))

    ax_index = None
    if len(parts) >= 2 and is_h_family_token(parts[0]) and is_ax_token(parts[1]):
        ax_index = 1
    elif is_ax_token(parts[0]):
        ax_index = 0
    elif is_ax_token(parts[-1]):
        ax_index = len(parts) - 1

    ax_token = parts[ax_index].upper() if ax_index is not None else "A0"
    skip_indices = set()
    if is_h_family_token(parts[0]):
        skip_indices.add(0)
    if ax_index is not None:
        skip_indices.add(ax_index)
    tail_index = len(parts) - 1
    if is_ax_token(parts[tail_index]) and tail_index != ax_index:
        skip_indices.add(tail_index)

    body_parts = [part for index, part in enumerate(parts) if index not in skip_indices]
    if not body_parts:
        return ax_token
    return f"{'-'.join(body_parts)}-{ax_token}"


def read_soc_label_table(label_xlsx: str | os.PathLike) -> dict[str, float]:
    """
    读取 SOC 标签表。
    English: read SOC label.

    输入 / Inputs:
    English: Inputs:.
    - `label_xlsx`: 至少包含 `SampleName` 和 SOC 数值列的 Excel 文件。
    English: - `label_xlsx`: `SampleName` SOC Excel file.

    输出 / Outputs:
    English: Outputs:.
    - `{SampleName: SOC}` 字典。
    English: - `{SampleName: SOC}` dictionary.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    df = pd.read_excel(label_xlsx)
    columns = [str(col).strip() for col in df.columns]
    if "SampleName" not in columns:
        raise ValueError(f"标签表缺少 SampleName 列，当前列为: {columns}")
    sample_col = df.columns[columns.index("SampleName")]
    soc_candidates = [col for col in df.columns if "soc" in str(col).lower()]
    if not soc_candidates:
        raise ValueError(f"标签表缺少 SOC 数值列，当前列为: {columns}")
    soc_col = soc_candidates[0]

    label_map: dict[str, float] = {}
    duplicate_names: set[str] = set()
    for _, row in df.iterrows():
        sample_name = str(row.get(sample_col, "")).strip()
        if not sample_name or sample_name.lower() == "nan":
            continue
        if sample_name in label_map:
            duplicate_names.add(sample_name)
            continue
        label_map[sample_name] = float(row[soc_col])
    if duplicate_names:
        raise ValueError(f"标签表存在重复 SampleName，数量: {len(duplicate_names)}")
    return label_map


def generate_public_sample_id(existing_ids: set[str], rng: random.Random | secrets.SystemRandom) -> str:
    """
    生成不重复的 8 位数字/字母随机样本名。
    English: 8 /sample.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    while True:
        sample_id = "".join(rng.choice(ALPHANUMERIC) for _ in range(8))
        if sample_id in existing_ids:
            continue
        if not any(ch.isdigit() for ch in sample_id):
            continue
        if not any(ch.isalpha() for ch in sample_id):
            continue
        existing_ids.add(sample_id)
        return sample_id


def resolve_default_cache_dir(source_root: str | os.PathLike | None) -> Path | None:
    """
    根据旧数据库根目录推断已校正缓存目录。
    English: database root directorycachedirectory.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    if source_root in (None, ""):
        return None
    candidate = Path(source_root) / DEFAULT_CACHE_SUBDIR
    return candidate if candidate.is_dir() else None


def iter_cache_records(cache_dir: str | os.PathLike, label_map: dict[str, float]) -> Iterable[dict[str, Any]]:
    """
    从旧训练磁盘缓存中迭代可公开样本记录。
    English: trainingcachesample.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    cache_root = Path(cache_dir)
    for sample_dir in sorted(path for path in cache_root.iterdir() if path.is_dir()):
        meta_path = sample_dir / "meta.json"
        hyper_path = sample_dir / "hyper.npy"
        nir_path = sample_dir / "nir.npy"
        image_path = sample_dir / "image.npy"
        if not (meta_path.is_file() and hyper_path.is_file() and nir_path.is_file() and image_path.is_file()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label_name = normalize_label_sample_name(meta.get("sample_name", ""))
        if label_name not in label_map:
            continue
        yield {
            "source": "calibrated_cache",
            "label_name": label_name,
            "soc": float(label_map[label_name]),
            "hyper_path": hyper_path,
            "nir_path": nir_path,
            "image_path": image_path,
        }


def collect_raw_records(source_data_dir: str | os.PathLike, label_map: dict[str, float]) -> list[dict[str, Any]]:
    """
    扫描原始样本库并构建可公开样本记录。
    English: samplebuildsample.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    data_root = Path(source_data_dir)
    sample_records: dict[str, list[dict[str, Any]]] = {}
    for root, dirs, _ in os.walk(data_root):
        dirs.sort()
        blank_name = next((item for item in dirs if "blank" in item.lower()), None)
        if not blank_name:
            continue
        blank_path = Path(root) / blank_name
        if not check_sample_files(str(blank_path)):
            continue
        for dirname in dirs:
            if dirname == blank_name:
                continue
            core_id = extract_core_id(dirname)
            if not core_id:
                continue
            sample_path = Path(root) / dirname
            sample_records.setdefault(dirname, []).append({
                "sample_path": sample_path,
                "blank_path": blank_path,
                "label_name": normalize_label_sample_name(dirname),
            })

    records: list[dict[str, Any]] = []
    for _, candidates in sample_records.items():
        if len(candidates) != 1:
            continue
        record = candidates[0]
        if record["label_name"] not in label_map:
            continue
        if not check_sample_files(str(record["sample_path"])):
            continue
        record["source"] = "raw_database"
        record["soc"] = float(label_map[record["label_name"]])
        records.append(record)
    return sorted(records, key=lambda item: item["label_name"])


def load_arrays_from_record(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取一个待公开样本的校正后多源数据。
    English: readsample.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    if record["source"] == "calibrated_cache":
        hyper = np.load(record["hyper_path"], allow_pickle=False).astype(np.float32, copy=False)
        nir = np.load(record["nir_path"], allow_pickle=False).astype(np.float32, copy=False)
        image = np.load(record["image_path"], allow_pickle=False).astype(np.float32, copy=False)
        return hyper, nir, image

    blank_path = record["blank_path"]
    sample_path = record["sample_path"]
    blank_hyper = read_hyper_csv(str(blank_path / "HyperVISNIR.csv"))
    sample_hyper = read_hyper_csv(str(sample_path / "HyperVISNIR.csv"))
    blank_nir = read_nir_csv(str(blank_path / "NIR.CSV"))
    sample_nir = read_nir_csv(str(sample_path / "NIR.CSV"))
    blank_image = read_images(str(blank_path))
    sample_image = read_images(str(sample_path))
    if blank_hyper is None or sample_hyper is None or blank_nir is None or sample_nir is None:
        raise ValueError("光谱或 NIR 读取失败。")
    if sample_image.shape != blank_image.shape:
        raise ValueError("样本图像与 blank 图像尺寸不一致。")
    hyper = sample_hyper / (blank_hyper + 1e-8)
    min_len = min(len(sample_nir), len(blank_nir))
    nir = sample_nir[:min_len] / (blank_nir[:min_len] + 1e-8)
    if len(nir) > 5:
        nir = nir[:5]
    elif len(nir) < 5:
        nir = np.concatenate([nir, np.zeros(5 - len(nir), dtype=np.float32)])
    image = np.transpose(sample_image / (blank_image + 1e-8), (2, 0, 1)).astype(np.float32)
    return hyper.astype(np.float32), nir.astype(np.float32), image.astype(np.float32)


def save_public_sample(
    output_path: Path,
    sample_id: str,
    hyper: np.ndarray,
    nir: np.ndarray,
    image: np.ndarray,
    soc_value: float,
    compressed: bool,
) -> None:
    """
    写出一个公开版 `.npz` 样本。
    English: public release `.npz` sample.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    writer = np.savez_compressed if compressed else np.savez
    writer(
        output_path,
        sample_id=np.asarray(sample_id),
        target_names=np.asarray(["SOC"]),
        labels=np.asarray([float(soc_value)], dtype=np.float32),
        hyper=np.asarray(hyper, dtype=np.float32),
        nir=np.asarray(nir, dtype=np.float32),
        image=np.asarray(image, dtype=np.float32),
    )


def prepare_output_root(output_root: str | os.PathLike, overwrite: bool) -> tuple[Path, Path]:
    """
    准备公开数据库输出目录。
    English: public databaseOutputdirectory.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    root = Path(output_root).expanduser()
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，请使用 --overwrite 或更换目录: {root}")
        shutil.rmtree(root)
    samples_dir = root / PUBLIC_SAMPLES_DIR_NAME
    samples_dir.mkdir(parents=True, exist_ok=True)
    return root, samples_dir


def write_manifest(output_root: Path, sample_count: int, compressed: bool, elapsed_seconds: float) -> None:
    """
    写出公开数据库清单。
    English: public database.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    manifest = {
        "database_format": PUBLIC_DATABASE_FORMAT,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "author": "ljy",
        "sample_count": int(sample_count),
        "samples_dir": PUBLIC_SAMPLES_DIR_NAME,
        "sample_file_extension": ".npz",
        "sample_id_rule": "8 uppercase letters/digits, at least one letter and one digit",
        "contains_original_sample_names": False,
        "contains_original_paths": False,
        "target_names": ["SOC"],
        "label_unit": {"SOC": "g/kg"},
        "hyper_dim": 681,
        "nir_dim": 5,
        "image_shape": [8, 1024, 1024],
        "image_channels": 8,
        "wavelengths": list(WAVELENGTHS),
        "blank_correction": "sample / (same-batch blank + 1e-8)",
        "npz_compressed": bool(compressed),
        "elapsed_seconds": float(elapsed_seconds),
    }
    (output_root / PUBLIC_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def rebuild_public_database(args: argparse.Namespace) -> Path:
    """
    执行公开数据库重构主流程。
    English: public database.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    start = time.time()
    labels = read_soc_label_table(args.label_xlsx)
    cache_dir = Path(args.cache_dir) if args.cache_dir else resolve_default_cache_dir(args.source_root)

    if cache_dir and cache_dir.is_dir():
        records = list(iter_cache_records(cache_dir, labels))
        source_mode = "calibrated_cache"
    else:
        if not args.source_data_dir:
            source_data_dir = Path(args.source_root) / "raw_samples"
        else:
            source_data_dir = Path(args.source_data_dir)
        records = collect_raw_records(source_data_dir, labels)
        source_mode = "raw_database"

    if args.limit is not None:
        records = records[: int(args.limit)]
    if not records:
        raise RuntimeError("没有找到可重构的有效样本。")

    output_root, samples_dir = prepare_output_root(args.output_root, overwrite=bool(args.overwrite))
    rng = random.Random(int(args.seed)) if args.seed is not None else secrets.SystemRandom()
    existing_ids: set[str] = set()
    written = 0
    failed = 0

    print(f">> 公开数据库重构开始，输入来源: {source_mode}")
    print(f">> 标签表有效记录数: {len(labels)}")
    print(f">> 待写出样本数: {len(records)}")
    print(f">> 输出目录: {output_root}")

    for index, record in enumerate(records, start=1):
        sample_id = generate_public_sample_id(existing_ids, rng)
        output_path = samples_dir / f"{sample_id}.npz"
        try:
            hyper, nir, image = load_arrays_from_record(record)
            save_public_sample(
                output_path=output_path,
                sample_id=sample_id,
                hyper=hyper,
                nir=nir,
                image=image,
                soc_value=float(record["soc"]),
                compressed=bool(args.compressed),
            )
            written += 1
        except Exception as error:
            failed += 1
            if output_path.exists():
                output_path.unlink()
            print(f"   [warning] 第 {index} 个样本写出失败，已跳过: {error}")
        if index == 1 or index % int(args.report_every) == 0 or index == len(records):
            print(f">> 进度: {index}/{len(records)} | 已写出: {written} | 失败: {failed}")

    elapsed = time.time() - start
    write_manifest(output_root, sample_count=written, compressed=bool(args.compressed), elapsed_seconds=elapsed)
    report = {
        "database_format": PUBLIC_DATABASE_FORMAT,
        "source_mode": source_mode,
        "label_count": int(len(labels)),
        "candidate_sample_count": int(len(records)),
        "written_sample_count": int(written),
        "failed_sample_count": int(failed),
        "contains_original_sample_names": False,
        "contains_original_paths": False,
        "elapsed_seconds": float(elapsed),
    }
    (output_root / "rebuild_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f">> 公开数据库重构完成: {output_root}")
    return output_root


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    English: parseparameter.

    最近修改时间 / Last modified: 2026-06-16
    English: Last modified: 2026-06-16.
    作者 / Author: ljy
    English: Author: ljy.
    """

    parser = argparse.ArgumentParser(description="Rebuild a public single-file soil sample database.")
    parser.add_argument("--source-root", default=None, help="旧数据库根目录；仅运行重构时提供。")
    parser.add_argument("--source-data-dir", default=None, help="旧原始样本库目录；未提供时由 source-root/raw_samples 推断。")
    parser.add_argument("--cache-dir", default=None, help="已校正旧缓存目录；未提供时由 source-root/SOC_SoilData/... 推断。")
    parser.add_argument("--label-xlsx", required=True, help="样本标签 Excel 文件，必须包含 SampleName 和 SOC 列。")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="公开数据库输出目录。")
    parser.add_argument("--compressed", action="store_true", help="使用 np.savez_compressed 压缩样本文件。")
    parser.add_argument("--overwrite", action="store_true", help="允许删除并重建已存在的输出目录。")
    parser.add_argument("--seed", type=int, default=None, help="匿名样本名随机种子；默认使用系统随机数。")
    parser.add_argument("--limit", type=int, default=None, help="只重构前 N 个样本，用于冒烟测试。")
    parser.add_argument("--report-every", type=int, default=50, help="每处理多少个样本输出一次进度。")
    return parser.parse_args()


def main() -> int:
    """
    命令行入口。
    English: Command-line entry point.
    """

    rebuild_public_database(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
