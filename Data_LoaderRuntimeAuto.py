# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Data_DiskCacheRegistry import DiskCacheRegistryManager, build_dataset_signature
from Data_PublicSampleDatabase import (
    is_public_sample_database,
    list_public_sample_files,
    load_public_sample_npz,
    read_public_manifest,
    target_names_for_mode,
)

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ================= 模块维护说明 =================
# EN: ================= maintainnote =================.
# 逻辑：
# EN: Logic:
# 1. 本文件负责训练期数据读取、同级 blank 校正、自动 memory/disk 缓存决策和样本级缓存读写；
# EN: this file train data loading, same blank, automatically memory/disk cache and sample cache read write;
# 2. Menu 文件不得直接读取数据，本文件由 Train_core 通过数据集构建入口调用；
# EN: Menu filemust notdirectlyreaddata, this file by Train_core pass data build interface use;
# 3. 最近注释维护：2026-05-29；作者：ljy。补充缓存决策、样本扫描和 Dataset 接口注释，不改变数据处理逻辑。
# EN: Latest comment maintenance: 2026-05-29; Author: ljy. cache, samplescan and Dataset interface, does not changedata logic.
# 4. 最近修改时间：2026-05-29；作者：ljy。读取进度改为终端动态单行刷新，避免每 1% 输出一行刷屏。
# EN: Last modified: 2026-05-29; Author: ljy.read degree change as terminal single line refresh, avoid each 1% line.
# 5. 最近修改时间：2026-06-16；作者：ljy。接入公开版单文件 .npz 数据库，训练期不再需要真实样本名或原始路径。
# EN: Last modified: 2026-06-16; Author: ljy. public release single file.npz data database, train no longer need need sample name or start path.


# ================= 固定波段配置 =================
# EN: ================= fixedbands =================.
WAVELENGTHS = ['0490', '0540', '0590', '0660', '0775', '0880', '0945', '1000']
IMAGE_CHANNELS = len(WAVELENGTHS)
VALID_ACTIVE_INPUTS = ("image", "hyper", "nir")


def normalize_active_inputs(active_inputs=None):
    """
    规范化当前训练实际启用的输入源。
    English: normalizecurrenttrainingInput.

    设计说明:
    English: Design note:
        该函数属于 Data 层的输入合同解析，确保训练库读取、缓存估算、blank 校正和 __getitem__
        English: Data Inputparse, trainingread, cache, blank __getitem__.
        与菜单 / 模型声明的 active_inputs 保持一致，避免 NIR-only 等任务在 CPU 端处理无关图像。
        English: 模型声明的 active_inputs 保持一致，避免 NIR-only 等任务在 CPU 端处理无关图像.
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
    """
    if active_inputs is None:
        return VALID_ACTIVE_INPUTS
    normalized = []
    for item in active_inputs:
        key = str(item).strip().lower()
        if not key:
            continue
        if key not in VALID_ACTIVE_INPUTS:
            raise ValueError(f"active_inputs 仅支持 {VALID_ACTIVE_INPUTS}，当前包含: {item!r}")
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized) or VALID_ACTIVE_INPUTS


def required_cache_files_for_inputs(active_inputs):
    """
    根据 active_inputs 返回当前样本缓存必须具备的 .npy 文件名。
    English: active_inputs returncurrentsamplecache .npy file.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    return tuple(f"{key}.npy" for key in normalize_active_inputs(active_inputs))


# ================= 辅助读取函数 =================
# EN: ================= readfunction =================.
def normalize_image_size(image_size):
    """
    统一解析图像尺寸配置。
    English: parseimageconfiguration.
    支持：
    English: :
    1. int，例如 224 -> (224, 224)
    English: 1. int, 224 -> (224, 224)
    2. tuple/list，例如 (1024, 1224) -> (1024, 1224)
    English: 2. tuple/list, (1024, 1224) -> (1024, 1224)
    3. None -> None，表示不做 resize
    English: 3. None -> None, resize.
    返回:
    English: return:
        None 或 (height, width)
        English: None (height, width)
    """
    if image_size is None:
        return None

    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError(f"image_size 必须为正整数，当前为: {image_size}")
        return (image_size, image_size)

    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        h, w = image_size
        h = int(h)
        w = int(w)
        if h <= 0 or w <= 0:
            raise ValueError(f"image_size 二维尺寸必须大于 0，当前为: {image_size}")
        return (h, w)

    raise ValueError(f"image_size 仅支持 int / (H, W) / None，当前为: {image_size}")


def parse_memory_limit_to_bytes(value):
    """
    把用户填写的内存预算解析为字节数。
    English: parse.
    支持：
    English: :
    1. int / float：默认按 GB 解释；
    English: float：默认按 GB 解释；.
    2. 字符串：支持 64、64GB、64000MB、8GiB 等写法；
    English: 2. : 64, 64GB, 64000MB, 8GiB ;
    3. None：返回 None，表示不参与自动决策。
    English: 3. None: return None, .
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(f"内存预算必须大于 0，当前为: {value}")
        return int(float(value) * (1024 ** 3))

    text = str(value).strip().upper().replace(" ", "")
    if not text:
        return None

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([A-Z]*)", text)
    if not match:
        raise ValueError(f"无法解析内存预算: {value}")

    number = float(match.group(1))
    unit = match.group(2) or "GB"
    unit_map = {
        "B": 1,
        "KB": 1000,
        "MB": 1000 ** 2,
        "GB": 1000 ** 3,
        "TB": 1000 ** 4,
        "KIB": 1024,
        "MIB": 1024 ** 2,
        "GIB": 1024 ** 3,
        "TIB": 1024 ** 4,
    }
    if unit not in unit_map:
        raise ValueError(f"不支持的内存单位: {unit}")
    return int(number * unit_map[unit])


def format_bytes(num_bytes: Optional[int]) -> str:
    """
    将字节数格式化为便于终端阅读的容量字符串。
    English: This docstring documents the corresponding function behavior and engineering constraints.

    输入:
    English: Input:
        num_bytes: 字节数；None 表示未知或未设置。
        English: num_bytes: ; None .
    输出:
    English: Output:
        形如 "12.34 GB" 的字符串。
        English: "12.34 GB" .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    if num_bytes is None:
        return "N/A"

    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{float(num_bytes):.2f} B"


def read_hyper_csv(filepath):
    """
    读取 HyperVISNIR.csv
    English: read HyperVISNIR.csv.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        values = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue

        if len(values) >= 681:
            return np.array(values[-681:], dtype=np.float32)
        return None
    except Exception as e:
        print(f"[Error] 读取 Hyper CSV 失败 {filepath}: {e}")
        return None


def read_nir_csv(filepath):
    """
    读取 NIR.CSV
    English: read NIR.CSV.
    """
    try:
        df = pd.read_csv(filepath, header=None)
        vals = df.values.flatten()
        vals = vals[~np.isnan(vals)]
        return vals.astype(np.float32)
    except Exception:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().replace(',', '\n')
                vals = [float(x) for x in content.split() if x.strip()]
            return np.array(vals, dtype=np.float32)
        except Exception as e2:
            print(f"[Error] 读取 NIR CSV 失败 {filepath}: {e2}")
            return None


def read_images(folder_path):
    """
    读取 8 张固定波长的 TIFF 图像。
    English: read 8 TIFF image.
    使用 np.fromfile + cv2.imdecode 兼容中文路径。
    English: np.fromfile + cv2.imdecode compatiblepath.
    """
    images = []

    for wl in WAVELENGTHS:
        fname = f"Image-{wl}nm.tif"
        fpath = os.path.join(folder_path, fname)

        if not os.path.exists(fpath):
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    if f.lower() == fname.lower():
                        fpath = os.path.join(folder_path, f)
                        break

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing image: {fname} in {folder_path}")

        try:
            img_array = np.fromfile(fpath, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            raise ValueError(f"Image decode failed: {fpath} ({e})")

        if img is None:
            raise ValueError(f"OpenCV 解码返回空值: {fpath}")

        images.append(img)

    img_stack = np.stack(images, axis=-1)
    return img_stack.astype(np.float32)


def extract_core_id(folder_name):
    """
    从样本文件夹名中提取 CoreID。
    English: sample file CoreID.
    规则：省略第一个 '-' 之前的前缀段，以及最后一个 '-' 之后的重复号。
    English: : '-' , '-' .
    """
    if not folder_name or '-' not in folder_name:
        return None

    parts = [part.strip() for part in folder_name.strip().split('-')]
    if len(parts) < 3:
        return None

    repeat_part = parts[-1]
    if not re.fullmatch(r"\d{4}", repeat_part):
        return None

    core_parts = parts[1:-1]
    if not core_parts or any(part == '' for part in core_parts):
        return None

    return '-'.join(core_parts)


def check_sample_files(folder_path, active_inputs=None):
    """
    仅检查训练所需关键文件是否存在，不做大文件读取。
    English: checktrainingfile, fileread.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    active_inputs = normalize_active_inputs(active_inputs)

    if "hyper" in active_inputs and not os.path.isfile(os.path.join(folder_path, "HyperVISNIR.csv")):
        return False
    if "nir" in active_inputs and not os.path.isfile(os.path.join(folder_path, "NIR.CSV")):
        return False

    if "image" in active_inputs:
        try:
            files_lower = {f.lower() for f in os.listdir(folder_path)}
        except Exception:
            return False

        for wl in WAVELENGTHS:
            fname = f"Image-{wl}nm.tif".lower()
            if fname not in files_lower:
                return False

    return True


DEFAULT_DISK_CACHE_DATABASE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ModelData", "DiskCache")


def get_default_disk_cache_root():
    """
    返回预处理磁盘缓存默认根目录。
    English: returncachedefaultdirectory.

    输出:
    English: Output:
        工程默认磁盘缓存目录路径。
        English: defaultcachedirectorypath.

    设计说明:
    English: Design note:
        缓存根目录与原始数据库放在同一大目录下，避免继续写入旧缓存目录。
        English: cachedirectorydirectory, avoidwritecachedirectory.
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
    """
    # 最近修改时间：2026-05-20；预处理磁盘缓存随原始数据库迁移到 磁盘，避免继续写入旧缓存目录。
    # EN: Last modified: 2026-05-20; disk cache start data database migrate to, avoid write old cache.
    return os.path.join(DEFAULT_DISK_CACHE_DATABASE_ROOT, "SOC_SoilData")


def format_image_size_tag(image_size):
    """
    将图像尺寸配置转为缓存目录标签。
    English: imageconfigurationcachedirectorylabel.

    输入:
    English: Input:
        image_size: None、int 或 (H, W)。
        English: image_size: None, int (H, W).
    输出:
    English: Output:
        "orig" 或 "HxW" 字符串。
        English: "orig" "HxW" .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    if image_size is None:
        return "orig"
    h, w = normalize_image_size(image_size)
    return f"{h}x{w}"


def sanitize_cache_name(name):
    """
    清洗样本名，使其可安全作为 Windows 文件夹名。
    English: sample, Windows file.

    输入:
    English: Input:
        name: 原始样本名或文件夹名。
        English: name: samplefile.
    输出:
    English: Output:
        替换非法路径字符后的缓存目录前缀。
        English: pathcachedirectory.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    name = str(name).strip()
    if not name:
        return "unknown_sample"
    return re.sub(r'[\\/:*?"<>|]+', '_', name)


def build_preprocess_cache_key(
    folder_path,
    blank_path,
    image_size,
    nir_dim,
    target_mode,
    image_channels=None,
    wavelengths=None,
):
    """
    构建单样本预处理缓存键。
    English: buildsamplecache.

    设计说明：
    English: Design note:
    1. 这里保留旧版哈希规则，只对函数签名做前向兼容扩展；
    English: 1. , compatible;
    2. 当前 dataset_loader 的调用方已经会传入 image_channels / wavelengths，
    English: wavelengths，.
       若这里仍使用旧签名，就会报 unexpected keyword argument；
       English: , unexpected keyword argument;
    3. 但为了避免 8 通道版本上线后把旧 sample 级缓存目录全部换名，
    English: 3. avoid 8 sample cachedirectory,.
       本函数暂不把 image_channels / wavelengths 实际写入 raw 哈希串；
       English: wavelengths 实际写入 raw 哈希串；.
    4. 通道数与波段配置的一致性，已经由数据库级 build_dataset_signature(...)
    English: 4. configuration, build_dataset_signature(...)
       在 cache registry 层负责，不需要再在 sample 级重复引入；
       English: cache registry , sample ;
    5. 这样既能修复当前报错，也不会无意义地触发整库 sample 缓存重建。
    English: 5. current, sample cache.
    """
    image_tag = format_image_size_tag(image_size)
    raw = "||".join([
        os.path.normpath(str(folder_path)).lower(),
        os.path.normpath(str(blank_path)).lower(),
        str(image_tag),
        str(int(nir_dim)),
        str(target_mode).lower().strip(),
    ])
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()


class DatasetReadProgressReporter:
    """
    训练数据库读取进度输出器。
    Training-database read progress reporter.

    逻辑 / Logic:
    English: Logic:.
    1. 以有效样本总数作为 100%，而不是以 blank 批次数作为 100%，便于直观看到样本级读取进度。
    English: 1. sample 100%, blank 100%, sampleread.
    2. `resolution_percent=1` 表示至少跨过 1 个百分点才输出一次，避免每个样本都刷屏。
    English: 2. `resolution_percent=1` 1 Output, avoidsample.
    3. 内存读取、磁盘缓存命中、磁盘缓存补建和失败跳过都调用 `advance()`，保证进度不会因坏样本或缺失缓存停住。
    English: 3. read, cache, cache `advance()`, ensuresamplemissingcache.
    4. 进度变化时使用同一终端行刷新，结束时只换行一次，避免输出栏持续增加。
    English: 4. , , avoidOutput.
    5. 最近修改时间 / Last modified: 2026-05-29；作者 / Author: ljy。
    English: Last modified: 2026-05-29；作者 / Author: ljy.
    """

    def __init__(self, total_samples, resolution_percent=1):
        """
        初始化样本级进度统计器。
        English: sample.

        输入:
        English: Input:
            total_samples: 本轮有效样本总数。
            English: total_samples: sample.
            resolution_percent: 至少跨过多少个百分点才输出一次进度。
            English: resolution_percent: Output.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        self.total_samples = max(int(total_samples), 0)
        self.resolution_percent = max(int(resolution_percent), 1)
        self.processed_samples = 0
        self.last_reported_percent = -1
        self.live_line_width = 0
        self.live_line_active = False

    def _write_live_line(self, text: str) -> None:
        """
        在同一终端行刷新读取进度。
        English: read.

        输入:
        English: Input:
            text: 当前要显示的完整进度文本。
            English: text: current.

        设计说明:
        English: Design note:
            使用回车符 `\r` 回到行首，并用空格覆盖上一条较长文本的尾部。
            English: `\r` , .
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        text = str(text)
        padding = " " * max(self.live_line_width - len(text), 0)
        sys.stdout.write("\r" + text + padding)
        sys.stdout.flush()
        self.live_line_width = len(text)
        self.live_line_active = True

    def _finish_live_line(self) -> None:
        """
        结束动态进度行并换行，避免下一条普通日志接在同一行后面。
        English: , avoid.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if self.live_line_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.live_line_width = 0
        self.live_line_active = False

    def start(self):
        """
        输出 0% 起始状态，使用动态单行显示。
        Print the 0% start state.

        最近修改时间 / Last modified: 2026-05-29；作者 / Author: ljy。
        English: Last modified: 2026-05-29；作者 / Author: ljy.
        """

        if self.total_samples <= 0:
            print(">> [Dataset] 数据读取进度: 无有效样本，跳过样本读取。")
            return

        self.last_reported_percent = 0
        self._write_live_line(f">> [Dataset] 数据读取进度: 000% | 0/{self.total_samples}")

    def advance(self, count=1, status="读取完成", sample_name=None):
        """
        推进读取进度，并按百分比粒度输出。
        Advance read progress and print at percentage granularity.

        最近修改时间 / Last modified: 2026-05-29；作者 / Author: ljy。
        English: Last modified: 2026-05-29；作者 / Author: ljy.
        """

        if self.total_samples <= 0:
            return

        self.processed_samples = min(
            self.total_samples,
            self.processed_samples + max(int(count), 0),
        )
        current_percent = int(self.processed_samples * 100 / self.total_samples)
        report_percent = (current_percent // self.resolution_percent) * self.resolution_percent
        if self.processed_samples >= self.total_samples:
            report_percent = 100

        if report_percent <= self.last_reported_percent:
            return

        self.last_reported_percent = report_percent
        sample_text = f" | 当前样本: {sample_name}" if sample_name else ""
        self._write_live_line(
            f">> [Dataset] 数据读取进度: {report_percent:03d}% | "
            f"{self.processed_samples}/{self.total_samples} | 状态: {status}{sample_text}"
        )

    def finish(self, status="读取完成"):
        """
        确保结束时输出 100%，并结束动态单行。
        Ensure that 100% is printed at the end.

        最近修改时间 / Last modified: 2026-05-29；作者 / Author: ljy。
        English: Last modified: 2026-05-29；作者 / Author: ljy.
        """

        if self.total_samples <= 0:
            return
        if self.last_reported_percent < 100:
            self.processed_samples = self.total_samples
            self.advance(count=0, status=status)
        self._finish_live_line()


class SoilMultiSourceDataset(Dataset):
    """
    SOC/TN 多源输入训练数据集。
    English: SOC/TN Inputtraining.

    输入源:
    English: Input:
        image: 8 通道固定波长图像，经同级 blank 校正后输出 [C, H, W]。
        English: image: 8 image, blank Output [C, H, W].
        hyper: 681 维 HyperVISNIR 特征，经同级 blank 校正。
        English: hyper: 681 HyperVISNIR , blank .
        nir: 指定维度 NIR 特征，经同级 blank 校正和截断/补零。
        English: nir: NIR , blank /.
        label: SOC、TN 或 SOC+TN 标签。
        English: label: SOC, TN SOC+TN label.

    缓存策略:
    English: cache:
        auto 模式先估算有效样本在目标分辨率下的内存占用；若超过预算则自动转 disk 模式。
        English: auto sample; disk .
        disk 模式下每个样本按 active_inputs 保存必要的 .npy / meta.json，并通过 registry 管理复用。
        English: meta.json，并通过 registry 管理复用.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    # ================= 数据集初始化与自动缓存决策 =================
    # EN: ================= data start and automaticallycache =================.
    # 设计目标：
    # EN: :
    # 1. 在完成“样本对齐 + 完整性筛选”之后，先估算目标分辨率下的数据集内存体量；
    # EN: in complete" + " after, first below data inside amount;
    # 2. 若估算体量落在用户给定预算内，则自动走 memory 模式；
    # EN: if amount in use inside, then automatically memory form;
    # 3. 若估算体量超出预算，则自动切换到 disk 模式，并把预处理结果缓存到 磁盘数据库目录下的 SOC_SoilData；
    # EN: if amount, then automatically to disk form, and result result cache to data database below SOC_SoilData;
    # 4. disk 模式下支持三种缓存策略：
    # EN: disk form below cache:
    #    - reuse_or_build：优先复用已有缓存，缺失时补建；
    # EN: reuse_or_build: preferreuse already cache, when;
    #    - reuse_only：优先读取 磁盘已有缓存，若缓存数据库损坏或数量不对，则自动删旧并重建；
    # EN: reuse_only: preferread already cache, if cachedata database or count not for, then automatically old and;
    #    - rebuild_all：忽略旧缓存，全部重建；
    # EN: rebuild_all: old cache, full;
    # 5. 新增 磁盘磁盘数据库管理文件：自动记录不同 target_mode / 分辨率 / NIR 维度对应的缓存目录；
    # EN: new data database file: automatically not same target_mode / / NIR degree for should cache;
    # 6. 下次训练时先通过管理文件定位历史数据库，再核对样本缓存数量，合法则直接复用；
    # EN: below train when first pass file data database, then for samplecachecount, then directlyreuse;
    # 7. 这样三份训练脚本就能共用同一套“先估算，再决策，再加载”的数据工程逻辑。
    # EN: train then can use same ", , " data program logic.
    # 8. 读取训练数据库时按有效样本数输出 1% 粒度进度，便于长时间加载时判断是否仍在推进。
    # EN: readtraindata database when by sample count 1% degree degree, to make it easier to when load when determine is no still in.
    def __init__(
        self,
        data_root,
        gt_path,
        image_size=None,
        nir_dim=5,
        target_mode='soc',
        tn_path=None,
        cache_mode='auto',
        cache_root=None,
        rebuild_cache=False,
        memory_limit=None,
        memory_utilization_ratio=0.85,
        memory_estimate_safety_factor=1.20,
        disk_cache_policy='reuse_or_build',
        write_cache_summary=True,
        cache_registry_enabled=True,
        cache_registry_filename='disk_cache_registry.json',
        active_inputs=None,
    ):
        """
        初始化数据集并完成样本扫描、缓存决策和缓存准备。
        English: sample, cachecache.

        输入:
        English: Input:
            data_root: 原始样本库根目录。
            English: data_root: sampledirectory.
            gt_path / tn_path: SOC/TN 真值文件路径。
            English: tn_path: SOC/TN 真值文件路径.
            image_size: 目标图像尺寸；None 表示保留原始尺寸。
            English: image_size: image; None .
            nir_dim: NIR 输出维度。
            target_mode: soc/tn/both。
            cache_mode: auto/memory/disk。
            cache_root: disk 模式缓存根目录；None 时使用 磁盘默认目录。
            English: cache_root: disk cachedirectory; None defaultdirectory.
            rebuild_cache / disk_cache_policy: 控制磁盘缓存复用或重建。
            English: disk_cache_policy: 控制磁盘缓存复用或重建.
            memory_limit 与相关比例: auto 模式的内存容量判定依据。
            English: memory_limit : auto .
            active_inputs: 当前模型实际启用输入源；None 表示 image/hyper/nir 全模态。
            English: active_inputs: currentmodelInput; None image/hyper/nir .

        输出:
        English: Output:
            构造完成后，self.data_cache 保存 memory 样本或 disk 样本索引。
            English: , self.data_cache save memory sample disk sample.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        self.data_root = data_root
        self.gt_path = gt_path
        self.tn_path = tn_path
        self.image_size = normalize_image_size(image_size)
        self.nir_dim = int(nir_dim)
        self.data_cache: List[Dict] = []
        self.active_inputs = normalize_active_inputs(active_inputs)
        self.required_cache_files = required_cache_files_for_inputs(self.active_inputs)
        self.target_mode = str(target_mode).lower().strip()
        self.cache_mode = str(cache_mode).lower().strip()
        self.rebuild_cache = bool(rebuild_cache)
        self.disk_cache_policy = str(disk_cache_policy).lower().strip()
        self.write_cache_summary = bool(write_cache_summary)
        self.cache_registry_enabled = bool(cache_registry_enabled)
        self.cache_registry_filename = str(cache_registry_filename).strip() if str(cache_registry_filename).strip() else 'disk_cache_registry.json'
        self.memory_limit_bytes = parse_memory_limit_to_bytes(memory_limit)
        self.memory_utilization_ratio = float(memory_utilization_ratio)
        self.memory_estimate_safety_factor = float(memory_estimate_safety_factor)
        self.cache_root = None
        self.cache_data_dir = None
        self.resolved_cache_mode = None
        self.cache_registry = None
        self.dataset_signature = None
        self.force_rebuild_due_to_registry = False
        self.registry_resolution_summary: Dict = {}
        self.cache_plan_summary: Dict = {}
        self.scan_summary: Dict = {}
        self.valid_records: List[Dict] = []
        self.group_records: List[Dict] = []

        if self.target_mode not in ('soc', 'tn', 'both'):
            raise ValueError(f"target_mode 仅支持 'soc' / 'tn' / 'both'，当前为: {target_mode}")

        if self.cache_mode not in ('auto', 'memory', 'disk'):
            raise ValueError(f"cache_mode 仅支持 'auto' / 'memory' / 'disk'，当前为: {cache_mode}")

        if self.disk_cache_policy not in ('reuse_or_build', 'reuse_only', 'rebuild_all'):
            raise ValueError(
                f"disk_cache_policy 仅支持 'reuse_or_build' / 'reuse_only' / 'rebuild_all'，当前为: {disk_cache_policy}"
            )

        # rebuild_cache=True 等价于强制全量重建。
        # EN: rebuild_cache=True full amount.
        if self.rebuild_cache:
            self.disk_cache_policy = 'rebuild_all'

        if is_public_sample_database(data_root):
            self._load_public_single_file_database(data_root=data_root)
            return

        print(">> [Dataset] 正在加载真值表...")
        self._load_ground_truths(gt_path=gt_path, tn_path=tn_path)

        print(">> [Dataset] 第一阶段：扫描样本目录、按目标模式对齐真值、检查完整性并排查重复样本...")
        scan_start = time.time()
        self._scan_and_build_valid_records(data_root=data_root)
        print(f">> [Dataset] 第一阶段完成，耗时: {time.time() - scan_start:.1f}s")

        self._resolve_cache_plan(cache_root=cache_root)
        self._prepare_dataset_cache()

    # ================= 公开版单文件数据库读取函数 =================
    # EN: ================= public release single filedata database readfunction =================.
    def _load_public_single_file_database(self, data_root):
        """
        读取公开版单文件样本数据库。
        English: readpublic releasesingle-file sample database.

        处理逻辑:
        English: Logic:
            1. 读取数据库根目录下的 `public_dataset_manifest.json`；
            English: 1. readdatabase root directory `public_dataset_manifest.json`;
            2. 将 `samples/*.npz` 作为训练样本索引，不读取原始路径或旧样本名；
            English: 2. `samples/*.npz` trainingsample, readpathsample;
            3. 样本数据在 `__getitem__` 中按需懒加载，避免初始化阶段一次性占用大量内存。
            English: 3. sample `__getitem__` load, avoid.

        最近修改时间：2026-06-16；作者：ljy。
        English: Last modified: 2026-06-16; Author: ljy.
        """

        manifest = read_public_manifest(data_root)
        available_targets = [str(item).upper() for item in manifest.get("target_names", ["SOC"])]
        requested_targets = target_names_for_mode(self.target_mode)
        missing_targets = [target for target in requested_targets if target not in available_targets]
        if missing_targets:
            raise ValueError(
                f"公开数据库缺少 TARGET_MODE={self.target_mode!r} 所需标签: {missing_targets}；"
                f"可用标签: {available_targets}"
            )

        sample_files = list_public_sample_files(data_root)
        self.resolved_cache_mode = "public_npz"
        self.cache_root = manifest["database_root"]
        self.cache_data_dir = str(Path(manifest["database_root"]) / "samples")
        self.valid_records = []
        self.data_cache = []
        for sample_file in sample_files:
            sample_id = sample_file.stem
            record = {
                "cache_mode": "public_npz",
                "sample_name": sample_id,
                "stable_split_id": sample_id,
                "core_id": sample_id,
                "folder_path": sample_id,
                "sample_file": str(sample_file),
            }
            self.valid_records.append(record)
            self.data_cache.append(record)

        self.scan_summary = {
            "database_format": manifest.get("database_format"),
            "target_mode": self.target_mode,
            "active_inputs": list(self.active_inputs),
            "valid_samples": int(len(self.data_cache)),
            "contains_original_sample_names": bool(manifest.get("contains_original_sample_names", False)),
            "contains_original_paths": bool(manifest.get("contains_original_paths", False)),
            "target_names": available_targets,
        }
        self.cache_plan_summary = {
            "requested_cache_mode": self.cache_mode,
            "resolved_cache_mode": self.resolved_cache_mode,
            "active_inputs": list(self.active_inputs),
            "required_cache_files": list(self.required_cache_files),
            "valid_samples": int(len(self.data_cache)),
            "final_dataset_length": int(len(self.data_cache)),
            "cache_root": self.cache_root,
            "cache_data_dir": self.cache_data_dir,
            "manifest_path": manifest.get("manifest_path"),
        }
        print("\n" + "=" * 60)
        print(" [公开版单文件数据库]")
        print(f"  数据库根目录: {self.cache_root}")
        print(f"  样本文件数: {len(self.data_cache)}")
        print(f"  实际输入源: {'+'.join(self.active_inputs)}")
        print(f"  当前目标模式: {self.target_mode}")
        print("=" * 60 + "\n")

    # ================= 真值加载函数 =================
    # EN: ================= value loadfunction =================.
    def _load_ground_truths(self, gt_path, tn_path=None):
        """
        读取 SOC/TN 真值表并构建 CoreID 到标签值的映射。
        English: read SOC/TN ground truthbuild CoreID label.

        输入:
        English: Input:
            gt_path: SOC.mat 路径，必须包含 SampleName 与 SOC_Value。
            English: gt_path: SOC.mat path, SampleName SOC_Value.
            tn_path: TN.mat 路径；target_mode 为 tn/both 时使用。
            English: tn_path: TN.mat path; target_mode tn/both .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        try:
            soc_data = sio.loadmat(gt_path)
            soc_sample_names = [str(x[0][0]).strip() for x in soc_data['SampleName']]
            soc_values = soc_data['SOC_Value'].flatten().astype(np.float32)
            self.soc_dict = dict(zip(soc_sample_names, soc_values))
            print(f"   SOC 真值加载完成，共 {len(self.soc_dict)} 条记录。")
        except Exception as e:
            raise ValueError(f"SOC 真值文件加载失败: {e}")

        self.tn_dict = {}
        if self.target_mode in ('tn', 'both'):
            if tn_path is None:
                tn_path = os.path.join(os.path.dirname(gt_path), "TN.mat")
            try:
                tn_data = sio.loadmat(tn_path)
                tn_sample_names = [str(x[0][0]).strip() for x in tn_data['SampleName']]
                tn_values = tn_data['TN_Value'].flatten().astype(np.float32)
                self.tn_dict = dict(zip(tn_sample_names, tn_values))
                print(f"   TN 真值加载完成，共 {len(self.tn_dict)} 条记录。")
            except Exception as e:
                raise ValueError(f"TN 真值文件加载失败: {e}")

    # ================= 扫描与样本筛选函数 =================
    # EN: ================= scan and sample function =================.
    def _scan_and_build_valid_records(self, data_root):
        """
        扫描原始样本目录并生成有效样本清单。
        English: sampledirectorysample.

        处理逻辑:
        English: Logic:
            1. 以包含 blank 的批次目录为基本单位；
            English: 1. blank directory;
            2. 按完整样本名去重，重复样本全部排除；
            English: 2. sample, sample;
            3. 依据 target_mode 检查 SOC/TN 真值是否可用；
            English: 3. target_mode check SOC/TN ground truth;
            4. 检查样本与 blank 的关键文件是否齐全。
            English: 4. checksample blank file.

        输出:
        English: Output:
            self.valid_records: 可进入训练/缓存的样本记录。
            English: self.valid_records: training/cachesample.
            self.group_records: 按 blank 批次聚合的样本组。
            English: self.group_records: blank sample.
            self.scan_summary: 扫描统计摘要。
            English: self.scan_summary: .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        fs_found_keys = set()
        scanned_groups = 0
        all_sample_name_records: Dict[str, List[Dict]] = {}

        for root, dirs, files in os.walk(data_root):
            dirs.sort()
            files.sort()
            blank_folder_name = None
            for d in dirs:
                if "blank" in d.lower():
                    blank_folder_name = d
                    break

            if not blank_folder_name:
                continue

            scanned_groups += 1
            blank_path = os.path.join(root, blank_folder_name)

            for d in dirs:
                if d == blank_folder_name:
                    continue

                sample_name = d.strip()
                core_id = extract_core_id(sample_name)
                if not core_id:
                    continue

                fs_found_keys.add(core_id)
                sample_folder_path = os.path.join(root, d)
                all_sample_name_records.setdefault(sample_name, []).append({
                    'root': root,
                    'blank_path': blank_path,
                    'folder_name': sample_name,
                    'folder_path': sample_folder_path,
                    'core_id': core_id,
                    'sample_name': sample_name,
                })

        duplicate_sample_names = {
            sample_name for sample_name, records in all_sample_name_records.items() if len(records) > 1
        }

        duplicate_report_items = []
        for sample_name in sorted(duplicate_sample_names):
            duplicate_report_items.extend(all_sample_name_records[sample_name])

        if duplicate_sample_names:
            print("\n" + "=" * 50)
            print(" [重复样本名称检查]")
            print(f" 发现重复完整样本名数量: {len(duplicate_sample_names)}")
            print(f" 被排除的重复样本目录总数: {len(duplicate_report_items)}")
            print(" 下面列出全部重复样本：")
            for item in duplicate_report_items:
                print(
                    f"   - SampleName: {item['sample_name']} | CoreID: {item['core_id']} | Path: {item['folder_path']}"
                )
            print("=" * 50 + "\n")
        else:
            print(">> [Dataset] 未发现重复完整样本名。")

        candidate_group_dict: Dict[str, Dict] = {}
        missing_file_records = 0
        invalid_label_records = 0

        for sample_name, records in all_sample_name_records.items():
            if sample_name in duplicate_sample_names:
                continue

            for record in records:
                core_id = record['core_id']
                has_soc = core_id in self.soc_dict
                has_tn = core_id in self.tn_dict if self.target_mode in ('tn', 'both') else False

                if self.target_mode == 'soc':
                    if not has_soc:
                        invalid_label_records += 1
                        continue
                    label = np.float32(self.soc_dict[core_id])
                elif self.target_mode == 'tn':
                    if not has_tn:
                        invalid_label_records += 1
                        continue
                    label = np.float32(self.tn_dict[core_id])
                else:
                    if not (has_soc and has_tn):
                        invalid_label_records += 1
                        continue
                    label = np.array([self.soc_dict[core_id], self.tn_dict[core_id]], dtype=np.float32)

                if not check_sample_files(record['folder_path'], active_inputs=self.active_inputs):
                    missing_file_records += 1
                    continue
                if not check_sample_files(record['blank_path'], active_inputs=self.active_inputs):
                    missing_file_records += 1
                    continue

                blank_path = record['blank_path']
                candidate_group_dict.setdefault(blank_path, {
                    'blank_path': blank_path,
                    'samples': [],
                })
                candidate_group_dict[blank_path]['samples'].append({
                    'folder_path': record['folder_path'],
                    'label': label,
                    'core_id': core_id,
                    'sample_name': sample_name,
                    'blank_path': blank_path,
                    'root': record['root'],
                })

        candidate_groups = list(candidate_group_dict.values())
        valid_records = []
        for group in candidate_groups:
            valid_records.extend(sorted(group['samples'], key=lambda x: x['sample_name']))

        print("\n" + "=" * 50)
        print(" [匹配完整性检查]")
        print(f"  当前目标模式: {self.target_mode}")
        print(f"  扫描到的 blank 批次数: {scanned_groups}")
        print(f"  扫描到的完整样本名总数: {len(all_sample_name_records)}")
        print(f"  重复完整样本名数量: {len(duplicate_sample_names)}")
        print(f"  因真值不满足要求被排除的样本数: {invalid_label_records}")
        print(f"  因关键文件不完整被排除的样本数: {missing_file_records}")
        print(f"  最终有效样本数: {len(valid_records)}")
        print("=" * 50 + "\n")

        self.valid_records = valid_records
        self.group_records = candidate_groups
        self.scan_summary = {
            'target_mode': self.target_mode,
            'active_inputs': list(self.active_inputs),
            'required_cache_files': list(self.required_cache_files),
            'scanned_groups': int(scanned_groups),
            'total_sample_name_records': int(len(all_sample_name_records)),
            'duplicate_sample_names': int(len(duplicate_sample_names)),
            'duplicate_report_items': int(len(duplicate_report_items)),
            'invalid_label_records': int(invalid_label_records),
            'missing_file_records': int(missing_file_records),
            'valid_samples': int(len(valid_records)),
            'fs_found_core_ids': int(len(fs_found_keys)),
        }

    # ================= 自动缓存模式决策函数 =================
    # EN: ================= automaticallycache form function =================.
    def _resolve_cache_plan(self, cache_root=None):
        """
        根据样本数量、目标分辨率和内存预算确定 memory/disk 缓存模式。
        English: sample, memory/disk cache.

        输入:
        English: Input:
            cache_root: 调用方指定的磁盘缓存根目录；None 时使用默认 磁盘缓存目录。
            English: cache_root: cachedirectory; None default cachedirectory.

        输出:
        English: Output:
            更新 self.resolved_cache_mode、self.cache_data_dir 和 self.cache_plan_summary。
            English: update self.resolved_cache_mode, self.cache_data_dir self.cache_plan_summary.

        设计说明:
        English: Design note:
            该函数只做容量估算和路径决策，真正的样本预处理由 _prepare_dataset_cache 完成。
            English: path, sample _prepare_dataset_cache .
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        if len(self.valid_records) == 0:
            self.resolved_cache_mode = 'memory' if self.cache_mode != 'disk' else 'disk'
            self.cache_plan_summary = {
                'requested_cache_mode': self.cache_mode,
                'resolved_cache_mode': self.resolved_cache_mode,
                'active_inputs': list(self.active_inputs),
                'required_cache_files': list(self.required_cache_files),
                'valid_samples': 0,
                'estimated_bytes_before_safety_factor': 0,
                'estimated_bytes_after_safety_factor': 0,
                'memory_limit_bytes': self.memory_limit_bytes,
                'memory_utilization_ratio': self.memory_utilization_ratio,
                'memory_estimate_safety_factor': self.memory_estimate_safety_factor,
            }
            return

        estimate_hw = self.image_size
        if "image" in self.active_inputs and estimate_hw is None:
            estimate_hw = self._infer_estimation_hw_from_first_valid_sample()

        per_sample_image_bytes = int(IMAGE_CHANNELS * estimate_hw[0] * estimate_hw[1] * 4) if "image" in self.active_inputs else 0
        per_sample_hyper_bytes = int(681 * 4) if "hyper" in self.active_inputs else 0
        per_sample_nir_bytes = int(self.nir_dim * 4) if "nir" in self.active_inputs else 0
        per_sample_label_bytes = 8 if self.target_mode == 'both' else 4
        per_sample_total_bytes = int(per_sample_image_bytes + per_sample_hyper_bytes + per_sample_nir_bytes + per_sample_label_bytes)
        estimated_total_bytes = int(len(self.valid_records) * per_sample_total_bytes)
        estimated_total_bytes_safe = int(estimated_total_bytes * self.memory_estimate_safety_factor)

        usable_memory_limit_bytes = None
        if self.memory_limit_bytes is not None:
            usable_memory_limit_bytes = int(self.memory_limit_bytes * self.memory_utilization_ratio)

        if self.cache_mode == 'memory':
            resolved_cache_mode = 'memory'
        elif self.cache_mode == 'disk':
            resolved_cache_mode = 'disk'
        else:
            if usable_memory_limit_bytes is None:
                resolved_cache_mode = 'disk'
            else:
                resolved_cache_mode = 'memory' if estimated_total_bytes_safe <= usable_memory_limit_bytes else 'disk'

        self.resolved_cache_mode = resolved_cache_mode

        if self.resolved_cache_mode == 'disk':
            self.cache_root = os.path.normpath(cache_root if cache_root else get_default_disk_cache_root())
            active_inputs_tag = "+".join(self.active_inputs)
            active_suffix = "" if self.active_inputs == VALID_ACTIVE_INPUTS else f"__inputs_{active_inputs_tag}"
            self.cache_data_dir = os.path.join(
                self.cache_root,
                f"img_{format_image_size_tag(self.image_size)}__nir_{int(self.nir_dim)}__{self.target_mode}__ch_{IMAGE_CHANNELS}{active_suffix}"
            )
            os.makedirs(self.cache_data_dir, exist_ok=True)
            self._resolve_disk_cache_dir_via_registry(self.cache_data_dir)

        self.cache_plan_summary = {
            'requested_cache_mode': self.cache_mode,
            'resolved_cache_mode': self.resolved_cache_mode,
            'disk_cache_policy': self.disk_cache_policy,
            'active_inputs': list(self.active_inputs),
            'required_cache_files': list(self.required_cache_files),
            'valid_samples': int(len(self.valid_records)),
            'estimate_image_hw': None if estimate_hw is None else [int(estimate_hw[0]), int(estimate_hw[1])],
            'image_channels': int(IMAGE_CHANNELS),
            'wavelengths': list(WAVELENGTHS),
            'per_sample_image_bytes': int(per_sample_image_bytes),
            'per_sample_hyper_bytes': int(per_sample_hyper_bytes),
            'per_sample_nir_bytes': int(per_sample_nir_bytes),
            'per_sample_label_bytes': int(per_sample_label_bytes),
            'per_sample_total_bytes': int(per_sample_total_bytes),
            'estimated_bytes_before_safety_factor': int(estimated_total_bytes),
            'estimated_bytes_after_safety_factor': int(estimated_total_bytes_safe),
            'memory_limit_bytes': self.memory_limit_bytes,
            'usable_memory_limit_bytes': usable_memory_limit_bytes,
            'memory_utilization_ratio': float(self.memory_utilization_ratio),
            'memory_estimate_safety_factor': float(self.memory_estimate_safety_factor),
            'formatted_estimated_bytes_before_safety_factor': format_bytes(estimated_total_bytes),
            'formatted_estimated_bytes_after_safety_factor': format_bytes(estimated_total_bytes_safe),
            'formatted_memory_limit': format_bytes(self.memory_limit_bytes),
            'formatted_usable_memory_limit': format_bytes(usable_memory_limit_bytes),
            'cache_root': self.cache_root,
            'cache_data_dir': self.cache_data_dir,
            'cache_registry_enabled': bool(self.cache_registry_enabled),
            'cache_registry_filename': self.cache_registry_filename,
            'dataset_signature': self.dataset_signature,
            'image_channels': int(IMAGE_CHANNELS),
            'wavelengths': list(WAVELENGTHS),
            'force_rebuild_due_to_registry': bool(self.force_rebuild_due_to_registry),
            'registry_resolution_summary': self.registry_resolution_summary,
        }

        print("\n" + "=" * 60)
        print(" [内存估算与缓存模式决策]")
        if estimate_hw is None:
            print("  目标分辨率: 未启用图像输入")
        else:
            print(f"  目标分辨率: {estimate_hw[0]} x {estimate_hw[1]}")
        print(f"  实际输入源: {'+'.join(self.active_inputs)}")
        print(f"  有效样本数: {len(self.valid_records)}")
        print(f"  估算单样本体量: {format_bytes(per_sample_total_bytes)}")
        print(f"  估算总量(未加安全系数): {format_bytes(estimated_total_bytes)}")
        print(f"  估算总量(加安全系数): {format_bytes(estimated_total_bytes_safe)}")
        print(f"  用户内存预算: {format_bytes(self.memory_limit_bytes)}")
        print(f"  可用于数据缓存的预算: {format_bytes(usable_memory_limit_bytes)}")
        print(f"  最终缓存模式: {self.resolved_cache_mode}")
        if self.resolved_cache_mode == 'disk':
            print(f"  磁盘缓存目录: {self.cache_data_dir}")
            print(f"  磁盘缓存策略: {self.disk_cache_policy}")
            if self.cache_registry_enabled:
                print(f"  数据库管理文件: {os.path.join(self.cache_root, self.cache_registry_filename)}")
                print(f"  数据库签名: {self.dataset_signature}")
                print(f"  管理文件命中旧库: {self.registry_resolution_summary.get('reuse_existing_cache', False)}")
                print(f"  管理文件要求重建: {self.force_rebuild_due_to_registry}")
        print("=" * 60 + "\n")

    def _infer_estimation_hw_from_first_valid_sample(self) -> Tuple[int, int]:
        """
        从第一个有效样本读取原始图像尺寸，用于 image_size=None 时估算内存。
        English: samplereadimage, image_size=None .

        输出:
            (height, width)。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if not self.valid_records:
            raise ValueError("当前没有有效样本，无法推断原始图像尺寸。")
        folder_path = self.valid_records[0]['folder_path']
        imgs = read_images(folder_path)
        h, w = imgs.shape[:2]
        return int(h), int(w)

    # ================= 磁盘磁盘数据库管理文件解析函数 =================
    # EN: ================= data database fileparsefunction =================.
    # 逻辑：
    # EN: Logic:
    # 1. 只有在最终决策为 disk 模式时，才启用该解析流程；
    # EN: only in most as disk form when, enable this parse program;
    # 2. 先基于“原始数据库路径 + 真值文件 + 目标模式 + 分辨率 + NIR 维度 + 图像通道配置”生成稳定 dataset_signature；
    # EN: first " + + + + NIR + "generatestable dataset_signature;
    # 3. 再通过 磁盘管理文件寻找该训练数据库对应的历史缓存目录，并核对 ready_sample_count；
    # EN: then pass file find this traindata database for should cache, and for ready_sample_count;
    # 4. 若找到合法旧数据库，则直接复用；若目录损坏或数量不一致，则自动删除旧目录并切到重建；
    # EN: if find to old data database, then directlyreuse; if or count not consistent, then automatically old and to;
    # 5. 这样用户无需手工维护缓存路径，脚本本身就能知道该读哪个历史数据库。
    # EN: use no need manualmaintaincachepath, then can this read data database.
    def _resolve_disk_cache_dir_via_registry(self, default_cache_data_dir):
        """
        通过磁盘缓存管理文件定位当前训练任务的缓存目录。
        English: cachefilecurrenttrainingcachedirectory.

        输入:
        English: Input:
            default_cache_data_dir: registry 没有合法记录时使用的默认目录。
            English: default_cache_data_dir: registry defaultdirectory.

        输出:
        English: Output:
            更新 self.cache_data_dir、self.dataset_signature、self.force_rebuild_due_to_registry。
            English: update self.cache_data_dir, self.dataset_signature, self.force_rebuild_due_to_registry.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        self.force_rebuild_due_to_registry = False
        self.registry_resolution_summary = {}

        if not self.cache_registry_enabled:
            self.cache_data_dir = default_cache_data_dir
            os.makedirs(self.cache_data_dir, exist_ok=True)
            return

        self.dataset_signature = build_dataset_signature(
            data_root=self.data_root,
            gt_path=self.gt_path,
            tn_path=self.tn_path,
            image_size=self.image_size,
            nir_dim=self.nir_dim,
            target_mode=self.target_mode,
            image_channels=IMAGE_CHANNELS,
            wavelengths=WAVELENGTHS,
            active_inputs=self.active_inputs,
        )
        self.cache_registry = DiskCacheRegistryManager(
            cache_root=self.cache_root,
            registry_filename=self.cache_registry_filename,
        )
        decision = self.cache_registry.resolve_cache_dir(
            dataset_signature=self.dataset_signature,
            default_cache_data_dir=default_cache_data_dir,
            expected_sample_count=len(self.valid_records),
            required_files=self.required_cache_files,
        )
        self.cache_data_dir = os.path.normpath(decision['cache_data_dir'])
        self.force_rebuild_due_to_registry = bool(decision.get('force_rebuild', False))
        self.registry_resolution_summary = decision

        if (
            self.active_inputs != VALID_ACTIVE_INPUTS
            and self.disk_cache_policy != 'rebuild_all'
            and not bool(decision.get('reuse_existing_cache', False))
        ):
            full_cache_data_dir = os.path.join(
                self.cache_root,
                f"img_{format_image_size_tag(self.image_size)}__nir_{int(self.nir_dim)}__{self.target_mode}__ch_{IMAGE_CHANNELS}",
            )
            full_cache_check = self.cache_registry.inspect_cache_dir(
                full_cache_data_dir,
                expected_sample_count=len(self.valid_records),
                required_files=self.required_cache_files,
            )
            full_cache_check['source'] = 'full_modal_superset'
            if full_cache_check['is_valid']:
                self.cache_data_dir = os.path.normpath(full_cache_data_dir)
                self.force_rebuild_due_to_registry = False
                self.registry_resolution_summary = {
                    **decision,
                    'cache_data_dir': self.cache_data_dir,
                    'reuse_existing_cache': True,
                    'force_rebuild': False,
                    'superset_cache_reuse': True,
                    'superset_cache_check': full_cache_check,
                    'selected_check': full_cache_check,
                }

    # ================= 磁盘数据库管理文件回写函数 =================
    # EN: ================= data database file write function =================.
    # 逻辑：
    # EN: Logic:
    # 1. 当 disk 数据集准备完成后，重新统计一次当前目录中真正就绪的样本缓存数；
    # EN: disk data complete after, new current in actually then samplecache number;
    # 2. 然后把目录路径、目标模式、样本数、完整性状态等回写到 磁盘管理文件；
    # EN: after path, form, sample count, complete property write to file;
    # 3. 这样下一次训练就可以直接通过管理文件命中历史数据库，而无需用户重新配置路径。
    # EN: below train then can directly pass file in data database, no need use new path.
    def _update_disk_cache_registry(self):
        """
        将当前 disk 缓存完整性状态回写到 registry。
        English: current disk cache registry.

        设计说明:
        English: Design note:
            仅在 disk 模式且 registry 启用时执行；memory 模式不产生磁盘数据库索引。
            English: disk registry ; memory .
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        if self.resolved_cache_mode != 'disk':
            return
        if not self.cache_registry_enabled:
            return
        if self.cache_registry is None or self.dataset_signature is None:
            return

        inspect_result = self.cache_registry.inspect_cache_dir(
            cache_data_dir=self.cache_data_dir,
            expected_sample_count=len(self.valid_records),
            required_files=self.required_cache_files,
        )
        self.cache_registry.update_entry(
            dataset_signature=self.dataset_signature,
            cache_data_dir=self.cache_data_dir,
            data_root=self.data_root,
            gt_path=self.gt_path,
            tn_path=self.tn_path,
            image_size=self.image_size,
            nir_dim=self.nir_dim,
            target_mode=self.target_mode,
            expected_sample_count=len(self.valid_records),
            ready_sample_count=inspect_result['ready_sample_count'],
            resolved_cache_mode=self.resolved_cache_mode,
            image_channels=IMAGE_CHANNELS,
            wavelengths=WAVELENGTHS,
            active_inputs=self.active_inputs,
            required_files=self.required_cache_files,
        )

    # ================= 数据预处理与缓存落盘函数 =================
    # EN: ================= data and cache function =================.
    def _prepare_dataset_cache(self):
        """
        按 blank 批次执行样本预处理，并填充 memory 缓存或 disk 缓存索引。
        English: blank sample, memory cache disk cache.

        处理逻辑:
        English: Logic:
            1. 每个 blank 批次只读取一次 blank 的 Hyper/NIR/Image；
            English: 1. blank read blank Hyper/NIR/Image;
            2. 对同批样本逐个执行 sample / blank 校正；
            English: blank 校正；.
            3. memory 模式直接把张量放入 self.data_cache；
            English: 3. memory self.data_cache;
            4. disk 模式先检查缓存命中，未命中时按策略新建 .npy 缓存；
            English: 4. disk checkcache, .npy cache;
            5. 按有效样本数输出读取进度，分辨率为 1%，缓存命中、补建、失败跳过都会推进。
            English: 5. sampleOutputread, 1%, cache, , .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        start_time = time.time()
        processed_groups = 0
        cache_hit_count = 0
        cache_build_count = 0
        cache_missing_count = 0
        read_progress_resolution_percent = 1
        progress_reporter = DatasetReadProgressReporter(
            total_samples=len(self.valid_records),
            resolution_percent=read_progress_resolution_percent,
        )
        progress_reporter.start()

        group_map: Dict[str, List[Dict]] = {}
        for sample in self.valid_records:
            group_map.setdefault(sample['blank_path'], []).append(sample)

        for blank_path in sorted(group_map.keys()):
            samples = sorted(group_map[blank_path], key=lambda x: x['sample_name'])

            if self.resolved_cache_mode == 'disk':
                samples_need_build = []

                for sample in samples:
                    cache_paths = self._build_disk_cache_paths(
                        folder_path=sample['folder_path'],
                        blank_path=blank_path,
                        sample_name=sample['sample_name']
                    )
                    cache_ready = self._is_disk_cache_ready(cache_paths)

                    if self.disk_cache_policy == 'rebuild_all':
                        samples_need_build.append((sample, cache_paths))
                        continue

                    if cache_ready:
                        self._append_disk_cache_record(
                            cache_paths=cache_paths,
                            label=sample['label'],
                            sample_name=sample['sample_name'],
                            core_id=sample['core_id'],
                            folder_path=sample['folder_path'],
                        )
                        cache_hit_count += 1
                        progress_reporter.advance(status="复用磁盘缓存", sample_name=sample['sample_name'])
                    else:
                        if self.disk_cache_policy == 'reuse_only':
                            cache_missing_count += 1
                            progress_reporter.advance(status="缓存缺失跳过", sample_name=sample['sample_name'])
                            continue
                        samples_need_build.append((sample, cache_paths))

                if self.disk_cache_policy == 'reuse_only' and samples_need_build:
                    raise FileNotFoundError(
                        f"reuse_only 模式下存在 {len(samples_need_build)} 个缺失缓存样本。"
                        f"请先在 磁盘缓存目录中准备完整缓存，或改为 reuse_or_build。"
                    )

                if not samples_need_build:
                    processed_groups += 1
                    continue

                try:
                    b_hyper = read_hyper_csv(os.path.join(blank_path, "HyperVISNIR.csv")) if "hyper" in self.active_inputs else None
                    b_nir = read_nir_csv(os.path.join(blank_path, "NIR.CSV")) if "nir" in self.active_inputs else None
                    b_imgs = read_images(blank_path) if "image" in self.active_inputs else None
                    if "hyper" in self.active_inputs and (b_hyper is None or len(b_hyper) != 681):
                        progress_reporter.advance(count=len(samples_need_build), status="blank Hyper 无效跳过")
                        continue
                    if "nir" in self.active_inputs and b_nir is None:
                        progress_reporter.advance(count=len(samples_need_build), status="blank NIR 无效跳过")
                        continue
                except Exception:
                    progress_reporter.advance(count=len(samples_need_build), status="blank 读取失败跳过")
                    continue

                for sample, cache_paths in samples_need_build:
                    ok = self._process_sample(
                        folder_path=sample['folder_path'],
                        label=sample['label'],
                        b_hyper=b_hyper,
                        b_nir=b_nir,
                        b_imgs=b_imgs,
                        sample_name=sample['sample_name'],
                        core_id=sample['core_id'],
                        cache_paths=cache_paths,
                    )
                    if ok:
                        self._append_disk_cache_record(
                            cache_paths=cache_paths,
                            label=sample['label'],
                            sample_name=sample['sample_name'],
                            core_id=sample['core_id'],
                            folder_path=sample['folder_path'],
                        )
                        cache_build_count += 1
                        progress_reporter.advance(status="新建磁盘缓存", sample_name=sample['sample_name'])
                    else:
                        progress_reporter.advance(status="样本读取失败跳过", sample_name=sample['sample_name'])

                processed_groups += 1
                continue

            try:
                b_hyper = read_hyper_csv(os.path.join(blank_path, "HyperVISNIR.csv")) if "hyper" in self.active_inputs else None
                b_nir = read_nir_csv(os.path.join(blank_path, "NIR.CSV")) if "nir" in self.active_inputs else None
                b_imgs = read_images(blank_path) if "image" in self.active_inputs else None
                if "hyper" in self.active_inputs and (b_hyper is None or len(b_hyper) != 681):
                    progress_reporter.advance(count=len(samples), status="blank Hyper 无效跳过")
                    continue
                if "nir" in self.active_inputs and b_nir is None:
                    progress_reporter.advance(count=len(samples), status="blank NIR 无效跳过")
                    continue
            except Exception:
                progress_reporter.advance(count=len(samples), status="blank 读取失败跳过")
                continue

            for sample in samples:
                ok = self._process_sample(
                    folder_path=sample['folder_path'],
                    label=sample['label'],
                    b_hyper=b_hyper,
                    b_nir=b_nir,
                    b_imgs=b_imgs,
                    sample_name=sample['sample_name'],
                    core_id=sample['core_id'],
                )
                if ok:
                    progress_reporter.advance(status="读取到内存", sample_name=sample['sample_name'])
                else:
                    progress_reporter.advance(status="样本读取失败跳过", sample_name=sample['sample_name'])

            processed_groups += 1

        progress_reporter.finish(status="读取完成")
        print(f">> ✅ 数据集加载完成！耗时: {time.time() - start_time:.1f}s")
        print(f"   处理批次: {processed_groups} | 有效样本数: {len(self.data_cache)}")
        if self.resolved_cache_mode == 'disk':
            print(f"   磁盘缓存命中样本数: {cache_hit_count}")
            print(f"   新建缓存样本数: {cache_build_count}")
            print(f"   缺失缓存样本数: {cache_missing_count}")
            print(f"   磁盘缓存目录: {self.cache_data_dir}")
        gc.collect()

        self.cache_plan_summary.update({
            'processed_groups': int(processed_groups),
            'cache_hit_count': int(cache_hit_count),
            'cache_build_count': int(cache_build_count),
            'cache_missing_count': int(cache_missing_count),
            'final_dataset_length': int(len(self.data_cache)),
            'read_progress_resolution_percent': int(read_progress_resolution_percent),
            'read_progress_total_samples': int(len(self.valid_records)),
        })

        if self.resolved_cache_mode == 'disk':
            self._update_disk_cache_registry()
            self.cache_plan_summary['registry_resolution_summary'] = self.registry_resolution_summary

        if self.write_cache_summary:
            self._write_runtime_summary_file()

    def _write_runtime_summary_file(self):
        """
        将本次扫描和缓存决策摘要写入 cache_runtime_summary.json。
        English: cachewrite cache_runtime_summary.json.

        说明:
        English: :
            该文件用于复盘缓存是否复用、样本数是否一致以及最终数据集长度。
            English: filecache, sample.
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        summary = {
            'scan_summary': self.scan_summary,
            'cache_plan_summary': self.cache_plan_summary,
        }
        if self.cache_data_dir is not None:
            os.makedirs(self.cache_data_dir, exist_ok=True)
            with open(os.path.join(self.cache_data_dir, 'cache_runtime_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    # ================= 磁盘缓存路径与记录函数 =================
    # EN: ================= disk cachepath and function =================.
    def _build_disk_cache_paths(self, folder_path, blank_path, sample_name=None):
        """
        构建单样本磁盘缓存目录和必要文件路径。
        English: buildsamplecachedirectoryfilepath.

        输入:
        English: Input:
            folder_path: 样本目录。
            English: folder_path: sampledirectory.
            blank_path: 同级 blank 目录。
            English: blank_path: blank directory.
            sample_name: 完整样本名，用于提高缓存目录可读性。
            English: sample_name: sample, cachedirectory.
        输出:
        English: Output:
            包含 cache_key、sample_cache_dir、hyper/nir/image/meta 路径的字典。
            English: cache_key, sample_cache_dir, hyper/nir/image/meta pathdictionary.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        cache_key = build_preprocess_cache_key(
            folder_path=folder_path,
            blank_path=blank_path,
            image_size=self.image_size,
            nir_dim=self.nir_dim,
            target_mode=self.target_mode,
            image_channels=IMAGE_CHANNELS,
            wavelengths=WAVELENGTHS,
        )
        safe_name = sanitize_cache_name(sample_name if sample_name is not None else os.path.basename(folder_path))
        sample_cache_dir = os.path.join(self.cache_data_dir, f"{safe_name}__{cache_key[:16]}")
        return {
            'cache_key': cache_key,
            'sample_cache_dir': sample_cache_dir,
            'hyper_path': os.path.join(sample_cache_dir, 'hyper.npy'),
            'nir_path': os.path.join(sample_cache_dir, 'nir.npy'),
            'image_path': os.path.join(sample_cache_dir, 'image.npy'),
            'meta_path': os.path.join(sample_cache_dir, 'meta.json'),
        }

    def _is_disk_cache_ready(self, cache_paths):
        """
        判断单个样本的 disk 缓存是否具备最小可读文件。
        English: determinesample disk cachefile.

        输入:
        English: Input:
            cache_paths: _build_disk_cache_paths 返回的路径字典。
            English: cache_paths: _build_disk_cache_paths returnpathdictionary.
        输出:
        English: Output:
            True 表示当前 active_inputs 需要的 .npy 均存在。
            English: True current active_inputs .npy .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return all(os.path.isfile(cache_paths[f"{key}_path"]) for key in self.active_inputs)

    def _append_disk_cache_record(self, cache_paths, label, sample_name, core_id, folder_path):
        """
        将一个已就绪的 disk 缓存样本登记到 Dataset 索引。
        English: disk cachesample Dataset .

        输入:
        English: Input:
            cache_paths: 样本缓存文件路径。
            English: cache_paths: samplecachefilepath.
            label: 训练标签。
            English: label: traininglabel.
            sample_name / core_id / folder_path: 后续输出和溯源所需元信息。
            English: core_id / folder_path: 后续输出和溯源所需元信息.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        record = {
            'cache_mode': 'disk',
            'label': np.asarray(label, dtype=np.float32),
            'sample_name': sample_name if sample_name is not None else os.path.basename(folder_path),
            'core_id': core_id if core_id is not None else extract_core_id(os.path.basename(folder_path)),
            'folder_path': folder_path,
        }
        for key in self.active_inputs:
            record[f"{key}_path"] = cache_paths[f"{key}_path"]
        self.data_cache.append(record)

    # ================= 单样本 blank 校正与缓存写入函数 =================
    # EN: ================= single sample blank and cachewritefunction =================.
    def _process_sample(self, folder_path, label, b_hyper, b_nir, b_imgs, sample_name=None, core_id=None, cache_paths=None):
        """
        对单个样本执行同级 blank 校正，并写入 memory 或 disk 缓存。
        English: sample blank , write memory disk cache.

        输入:
        English: Input:
            folder_path: 样本目录。
            English: folder_path: sampledirectory.
            label: 当前 target_mode 下的标签。
            English: label: current target_mode label.
            b_hyper / b_nir / b_imgs: 同批 blank 的光谱、NIR 和图像。
            English: b_nir / b_imgs: 同批 blank 的光谱、NIR 和图像.
            sample_name / core_id: 样本溯源信息。
            English: core_id: 样本溯源信息.
            cache_paths: disk 模式下的输出路径。
            English: cache_paths: disk Outputpath.
        输出:
        English: Output:
            True 表示样本处理成功；False 表示读取、维度或保存失败。
            English: True sample; False read, save.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        try:
            hyper_cal = None
            nir_cal = None
            img_tensor = None

            if "hyper" in self.active_inputs:
                s_hyper = read_hyper_csv(os.path.join(folder_path, "HyperVISNIR.csv"))
                if s_hyper is None or len(s_hyper) != 681 or b_hyper is None:
                    return False
                hyper_cal = s_hyper / (b_hyper + 1e-8)

            if "nir" in self.active_inputs:
                s_nir = read_nir_csv(os.path.join(folder_path, "NIR.CSV"))
                if s_nir is None or b_nir is None:
                    return False

                min_len = min(len(s_nir), len(b_nir))
                s_nir_cut = s_nir[:min_len]
                b_nir_cut = b_nir[:min_len]
                nir_cal = s_nir_cut / (b_nir_cut + 1e-8)

                if len(nir_cal) > self.nir_dim:
                    nir_cal = nir_cal[:self.nir_dim]
                elif len(nir_cal) < self.nir_dim:
                    pad = np.zeros(self.nir_dim - len(nir_cal), dtype=np.float32)
                    nir_cal = np.concatenate([nir_cal, pad])

            if "image" in self.active_inputs:
                s_imgs = read_images(folder_path)
                if b_imgs is None or s_imgs.shape != b_imgs.shape:
                    return False

                img_cal = s_imgs / (b_imgs + 1e-8)
                img_np = np.transpose(img_cal, (2, 0, 1)).astype(np.float32)
                img_tensor_raw = torch.from_numpy(img_np).unsqueeze(0)

                if self.image_size is None:
                    img_tensor = img_tensor_raw.squeeze(0)
                else:
                    img_tensor = F.interpolate(
                        img_tensor_raw,
                        size=self.image_size,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)

            if self.resolved_cache_mode == 'disk':
                if cache_paths is None:
                    raise ValueError("disk 模式下 _process_sample 必须提供 cache_paths。")

                os.makedirs(cache_paths['sample_cache_dir'], exist_ok=True)
                if "hyper" in self.active_inputs:
                    np.save(cache_paths['hyper_path'], hyper_cal.astype(np.float32))
                if "nir" in self.active_inputs:
                    np.save(cache_paths['nir_path'], nir_cal.astype(np.float32))
                if "image" in self.active_inputs:
                    np.save(cache_paths['image_path'], img_tensor.cpu().numpy().astype(np.float32))
                with open(cache_paths['meta_path'], 'w', encoding='utf-8') as f:
                    json.dump({
                        'sample_name': sample_name if sample_name is not None else os.path.basename(folder_path),
                        'core_id': core_id if core_id is not None else extract_core_id(os.path.basename(folder_path)),
                        'folder_path': folder_path,
                        'active_inputs': list(self.active_inputs),
                        'required_cache_files': list(self.required_cache_files),
                        'image_size': None if self.image_size is None else list(self.image_size),
                        'image_channels': int(IMAGE_CHANNELS),
                        'wavelengths': list(WAVELENGTHS),
                        'nir_dim': int(self.nir_dim),
                        'target_mode': self.target_mode,
                    }, f, ensure_ascii=False, indent=2)
                return True

            record = {
                'label': torch.tensor(label, dtype=torch.float32),
                'sample_name': sample_name if sample_name is not None else os.path.basename(folder_path),
                'core_id': core_id if core_id is not None else extract_core_id(os.path.basename(folder_path)),
                'folder_path': folder_path,
            }
            if "hyper" in self.active_inputs:
                record['hyper'] = torch.from_numpy(hyper_cal)
            if "nir" in self.active_inputs:
                record['nir'] = torch.from_numpy(nir_cal)
            if "image" in self.active_inputs:
                record['image'] = img_tensor
            self.data_cache.append(record)
            return True

        except Exception:
            return False

    def get_cache_plan_summary(self):
        """
        返回缓存模式决策和执行统计摘要。
        English: returncache.

        输出:
        English: Output:
            self.cache_plan_summary 的浅拷贝，避免外部直接改写内部状态。
            English: self.cache_plan_summary , avoid.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return dict(self.cache_plan_summary)

    def __len__(self):
        """
        返回当前 Dataset 中可训练样本数量。
        English: returncurrent Dataset trainingsample.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return len(self.data_cache)

    def __getitem__(self, idx):
        """
        按索引读取一个训练样本。
        English: readtrainingsample.

        输入:
        English: Input:
            idx: 样本下标。
            English: idx: sample.
        输出:
        English: Output:
            包含 hyper、nir、image、label、sample_name、core_id、folder_path 的字典。
            English: hyper, nir, image, label, sample_name, core_id, folder_path dictionary.

        说明:
        English: :
            memory 模式直接返回内存中的张量；disk 模式按路径延迟加载 .npy，降低常驻内存占用。
            English: memory return; disk pathload .npy, .
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        item = self.data_cache[idx]

        if item.get('cache_mode') == 'public_npz':
            return load_public_sample_npz(
                sample_path=item['sample_file'],
                active_inputs=self.active_inputs,
                image_size=self.image_size,
                nir_dim=self.nir_dim,
                target_mode=self.target_mode,
            )

        if item.get('cache_mode', 'memory') != 'disk':
            return item

        batch_item = {
            'label': torch.tensor(item['label'], dtype=torch.float32),
            'sample_name': item['sample_name'],
            'core_id': item['core_id'],
            'folder_path': item['folder_path'],
        }
        for key in self.active_inputs:
            batch_item[key] = torch.from_numpy(np.load(item[f'{key}_path'], allow_pickle=False).astype(np.float32, copy=False).copy())
        return batch_item


