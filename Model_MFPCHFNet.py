# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
# =============================================================================
# Model_MFPCHFNet.py
# MultiFrequency Principal Component  Hierarchical Fusion Network
#
# 中文名称：
# EN: in name:
#   主成分层级融合网络
# EN: principal components fusion.
#
# 本文件为 SOC_MFPCHFNet 的工程 V2 版本。
# EN: this file as SOC_MFPCHFNet program V2.
# 重要约定：V2 仅表示代码工程版本；论文中仍统一命名为 MFPC-HFNet，不标注 V2。
# EN: need: V2 onlymeans code program; in still name as MFPC-HFNet, not V2.
#
# 最近修改时间：2026-05-29。
# EN: Last modified: 2026-05-29.
#
# 主要拓扑：
# EN: main topology:
# 1. 输入图像 [B, 8, 1024, 1024] 先构建 3 层拉普拉斯金字塔。
# EN: image [B, 8, 1024, 1024] first build 3 Laplacian pyramid.
# 2. high1 / high2 / high3 / low 均从 pca_priors_full.pt 读取固定通道归一化参数
# EN: high1 / high2 / high3 / low from pca_priors_full.pt readfixedchannel-normalization parameters.
#    与固定 PCA 1×1 投影参数，保证离线先验、训练、推理三端一致。
# EN: and fixed PCA 1x1 projection parameters, ensureoffline priors, train, consistent.
# 3. high1 / high2 / high3 分别通过 PCASE 压缩为 64×64、32×32、16×16
# EN: high1 / high2 / high3 pass PCASE as 64x64, 32x32, 16x16.
#    source map。V2 将 PCASE 内部每个 PC 分支的连续下采样块替换为
# EN: source map.V2 PCASE inside each PC below as.
#    128×128 口径 EfficientNetV2-style 压缩结构；PCASE 容量公式、PCA
# EN: 128x128 interface EfficientNetV2-style result; PCASE amount form, PCA.
#    先验和后续 token/fusion 接口保持不变。
# EN: first and later token/fusion interface not.
# 4. high 高频摘要聚合采用单向 child-to-parent 路径：
# EN: high high-frequency summary use single child-to-parent path:
#    H1 -> PixelUnshuffle(2) -> H2 融合 -> CrossPatchEncoder -> HF2；
# EN: H1 -> PixelUnshuffle(2) -> H2 fusion -> CrossPatchEncoder -> HF2;
#    HF2 -> PixelUnshuffle(2) -> H3 融合 -> CrossPatchEncoder -> HF summary。
# EN: HF2 -> PixelUnshuffle(2) -> H3 fusion -> CrossPatchEncoder -> HF summary.
# 5. low 层级采用当前修正版：
# EN: low use current:
#    low PC map 通过 PCASE 直接压缩成 8×8 low source。由于 low 层不执行
# EN: low PC map pass PCASE directly 8x8 low source. by low not execute.
#    结构向量筛选，其有效特征向量保留率固定为 1.0。随后，8×8 low source
# EN: structural-vector selection, its amount keep fixed as 1.0. after, 8x8 low source.
#    作为一个整体局部块，由共享 LD 入口编码为 1×1 low token。
# EN: as local block, by LD interface programming code as 1x1 low token.
# 6. H3 -> low 的融合与 H2 -> H3 保持同一 child-to-parent 思路：
# EN: H3 -> low fusion and H2 -> H3 same child-to-parent path:
#    HF summary [B, Dhf, 2, 2] 经 PixelUnshuffle(2) 折叠为 1×1 父级摘要，
# EN: HF summary [B, Dhf, 2, 2] PixelUnshuffle(2) as 1x1 parent summary,.
#    再与 low token [B, Dlow, 1, 1] 融合并执行 CrossPatchEncoder。
# EN: then and low token [B, Dlow, 1, 1] fusion and execute CrossPatchEncoder.
# 7. token 宽度由 PCASE 后的 patch 源容量和固定压缩倍率自适应推导，
# EN: token degree by PCASE after patch source amount and fixedcompression ratio should,.
#    默认 token_compression_ratio=8，避免不同频层信息瓶颈不一致。
# EN: default token_compression_ratio=8, avoid not same frequency bandinformation bottleneck not consistent.
# 8. 最终保持现有训练接口 forward(hyper, nir, image)，便于替换旧模型文件。
# EN: most train interface forward(hyper, nir, image), to make it easier to old file.
# 9. 输入端消融由 active_inputs 控制，只实例化菜单声明的输入分支，保证续训
# EN: input-side ablation by active_inputs control, only single input branches, ensure.
#    checkpoint 中不存在未启用分支的无效参数。
# EN: checkpoint in does not exist not yet enable no parameters.
# 10. 最近注释维护：2026-05-29；作者：ljy。补充构造函数、forward 路径和工具函数
# EN: Latest comment maintenance: 2026-05-29; Author: ljy. constructor, forward path and utility functions.
#     的输入输出说明，不改变 MFPC-HFNet 参数、拓扑或训练接口。
# EN: inputs and outputsnote, does not change MFPC-HFNet parameters, or train interface.
#
# Notes:
# - PCASE is not LD Encoder.
# - 高频 LD Encoder 在同一频层内共享参数，跨频层独立配置。
# EN: high LD Encoder in same frequency band inside parameters, frequency band.
# - low 层不再使用 PACEBlock 上采样融合，而是作为 1×1 low token 接收高频摘要。
# EN: low no longeruse PACEBlock on fusion, is as 1x1 low token high-frequency summary.
# =============================================================================

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGE_CHANNELS = 8


# =============================================================================
# 1. Laplacian pyramid
# =============================================================================

def build_gaussian_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    构建与离线先验一致的固定 5×5 Gaussian kernel。
    Build the fixed 5×5 Gaussian kernel used by the Laplacian pyramid.
    """
    kernel = torch.tensor(
        [
            [1., 4., 6., 4., 1.],
            [4., 16., 24., 16., 4.],
            [6., 24., 36., 24., 6.],
            [4., 16., 24., 16., 4.],
            [1., 4., 6., 4., 1.],
        ],
        device=device,
        dtype=dtype,
    )
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, 5, 5).repeat(IMAGE_CHANNELS, 1, 1, 1)


def smooth_image(img_bchw: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    对 8 通道图像执行 depthwise Gaussian smoothing。
    English: 8 image depthwise Gaussian smoothing.
    """
    return F.conv2d(img_bchw, kernel, stride=1, padding=2, groups=IMAGE_CHANNELS)


def pyramid_down(img_bchw: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Gaussian smoothing + 2× downsampling."""
    smoothed = smooth_image(img_bchw, kernel)
    return smoothed[:, :, ::2, ::2]


def pyramid_up(img_bchw: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Upsample to target spatial size."""
    return F.interpolate(img_bchw, size=target_hw, mode="bilinear", align_corners=False)


def build_laplacian_pyramid(img_bchw: torch.Tensor, num_levels: int = 3) -> Dict[str, torch.Tensor]:
    """
    构建 3 层拉普拉斯金字塔。
    Build a 3-level Laplacian pyramid.

    Output:
        high1: [B, 8, 1024, 1024]
        high2: [B, 8, 512, 512]
        high3: [B, 8, 256, 256]
        low:   [B, 8, 128, 128]
    """
    if int(num_levels) != 3:
        raise ValueError(f"PC-HFNet 当前固定使用 3 个高频层级，num_levels={num_levels} 不支持。")
    if img_bchw.dim() != 4:
        raise ValueError(f"image 必须为 [B, C, H, W]，当前 shape={tuple(img_bchw.shape)}。")
    if img_bchw.shape[1] != IMAGE_CHANNELS:
        raise ValueError(f"图像输入固定为 {IMAGE_CHANNELS} 通道，当前为 {img_bchw.shape[1]}。")

    current = img_bchw
    kernel = build_gaussian_kernel(current.device, current.dtype)
    highs = []

    for _ in range(3):
        down = pyramid_down(current, kernel)
        up = pyramid_up(down, current.shape[-2:])
        highs.append(current - up)
        current = down

    return {"high1": highs[0], "high2": highs[1], "high3": highs[2], "low": current}


VALID_MFPCHF_STRUCTURE_BANDS = ("high1", "high2", "high3", "low")
SUPPORTED_MFPCHF_STRUCTURES = {
    "full": ("high1", "high2", "high3", "low"),
    "3hf": ("high1", "high2", "high3", "low"),
    "high1+high2+high3+low": ("high1", "high2", "high3", "low"),
    "h1+h2+h3+low": ("high1", "high2", "high3", "low"),
    "h2h3low": ("high2", "high3", "low"),
    "h2h3_low": ("high2", "high3", "low"),
    "2hf": ("high2", "high3", "low"),
    "high2+high3+low": ("high2", "high3", "low"),
    "h2+h3+low": ("high2", "high3", "low"),
    "h3low": ("high3", "low"),
    "h3_low": ("high3", "low"),
    "1hf": ("high3", "low"),
    "high3+low": ("high3", "low"),
    "h3+low": ("high3", "low"),
    "low": ("low",),
    "lowonly": ("low",),
    "low_only": ("low",),
}


def normalize_mfpchf_structure(structure: Optional[Union[str, Sequence[str]]]) -> Tuple[str, ...]:
    """
    将菜单传入的结构标签规范化为频层元组。
    Normalize menu structure labels to a tuple of active frequency bands.

    逻辑 / Logic:
    English: Logic:.
    1. 菜单只声明 Full、H2H3Low、H3Low 和 LowOnly 四类连续后缀结构；
    English: 1. menu Full, H2H3Low, H3Low LowOnly ;
    2. 模型层只按菜单标签构造对应频层，不在训练引擎中推断隐藏结构；
    English: 2. modelmenulabel, training engine;
    3. 最近修改时间：2026-05-29；作者：ljy。
    English: 3. Last modified: 2026-05-29; Author: ljy.
    """
    if structure is None or structure == "":
        return SUPPORTED_MFPCHF_STRUCTURES["full"]

    if isinstance(structure, (list, tuple)):
        bands = tuple(str(item).strip().lower() for item in structure if str(item).strip())
        if bands in set(SUPPORTED_MFPCHF_STRUCTURES.values()):
            return bands
        raise ValueError(f"不支持的 MFPC-HFNet 结构频层组合: {bands}。")

    key = str(structure).strip().lower()
    key = key.replace(" ", "")
    key = key.replace("-", "_")
    key = key.replace("/", "+")
    if key in SUPPORTED_MFPCHF_STRUCTURES:
        return SUPPORTED_MFPCHF_STRUCTURES[key]

    parts = tuple(part for part in key.split("+") if part)
    alias = {"h1": "high1", "h2": "high2", "h3": "high3"}
    bands = tuple(alias.get(part, part) for part in parts)
    if bands in set(SUPPORTED_MFPCHF_STRUCTURES.values()):
        return bands

    raise ValueError(
        "structure 仅支持 Full/H2H3Low/H3Low/LowOnly，"
        f"当前为 {structure!r}。"
    )


def build_structure_laplacian_pyramid(img_bchw: torch.Tensor, structure: Sequence[str]) -> Dict[str, torch.Tensor]:
    """
    按结构消融口径构建频层输入。
    Build the active pyramid bands for structural ablation.

    物理意义 / Physical meaning:
    English: Physical meaning:.
    - Full: 1024 输入生成 high1/high2/high3/low；
    English: - Full: 1024 Input high1/high2/high3/low;
    - H2H3Low: 512 输入生成 high2/high3/low；
    English: - H2H3Low: 512 Input high2/high3/low;
    - H3Low: 256 输入生成 high3/low；
    English: - H3Low: 256 Input high3/low;
    - LowOnly: 128 输入直接作为 low。
    English: - LowOnly: 128 Input low.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    bands = normalize_mfpchf_structure(tuple(structure))
    if img_bchw.dim() != 4:
        raise ValueError(f"image 必须为 [B, C, H, W]，当前 shape={tuple(img_bchw.shape)}。")
    if img_bchw.shape[1] != IMAGE_CHANNELS:
        raise ValueError(f"图像输入固定为 {IMAGE_CHANNELS} 通道，当前为 {img_bchw.shape[1]}。")

    high_bands = [band for band in bands if band != "low"]
    pyramid: Dict[str, torch.Tensor] = {}
    current = img_bchw
    if high_bands:
        kernel = build_gaussian_kernel(current.device, current.dtype)
        for band_name in high_bands:
            down = pyramid_down(current, kernel)
            up = pyramid_up(down, current.shape[-2:])
            pyramid[band_name] = current - up
            current = down

    pyramid["low"] = current
    return pyramid


# =============================================================================
# 2. PCA prior reading and fixed projection
# =============================================================================

def _load_torch_file(path: str) -> Dict:
    """
    兼容不同 PyTorch 版本的 torch.load。
    English: compatible PyTorch torch.load.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get_first_available(d: Dict, keys: List[str], band_name: str):
    """
    从频层先验字典中读取第一个存在的候选字段。
    English: dictionaryreadfield.

    输入:
    English: Input:
        d: 单个频层的 prior 字典。
        English: d: prior dictionary.
        keys: 兼容字段名候选列表。
        English: keys: compatiblefieldlist.
        band_name: 频层名，用于报错定位。
        English: band_name: , .
    输出:
    English: Output:
        命中的字段值。
        English: field.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    for key in keys:
        if key in d:
            return d[key]
    raise KeyError(f"pca_priors['{band_name}'] 缺少字段，候选名为 {keys}。")


def _get_optional_prior_float(d: Dict, keys: List[str], default=None):
    """
    从先验字典中读取可选 float 字段。
    Read an optional float field from prior dict.

    设计目的：
    English: Design purpose:
    1. 新版 pca_priors_full.pt 保存 effective_feature_vector_ratio；
    English: 1. pca_priors_full.pt save effective_feature_vector_ratio;
    2. 旧版先验文件可能没有该字段，因此这里必须保持兼容；
    English: 2. filefield, compatible;
    3. 若字段缺失，则返回 default，由模型构造参数作为兜底。
    English: 3. fieldmissing, return default, modelparameter.
    """
    for key in keys:
        if key not in d:
            continue

        value = d[key]
        if value is None:
            continue

        if torch.is_tensor(value):
            value = value.detach().cpu().flatten()[0].item()

        try:
            return float(value)
        except Exception:
            continue

    return default


def _normalize_prior_root(priors: Dict) -> Dict:
    """
    兼容可能的外层封装。
    Compatible with a possible outer wrapper in the prior file.
    """
    if not isinstance(priors, dict):
        raise TypeError("pca_priors_full.pt 读取结果必须为 dict。")

    for wrapper_key in ("pca_priors", "priors", "bands", "layers"):
        if wrapper_key in priors and isinstance(priors[wrapper_key], dict):
            candidate = priors[wrapper_key]
            if all(k in candidate for k in ("high1", "high2", "high3", "low")):
                return candidate

    return priors


class FixedChannelNormalizer(nn.Module):
    """
    固定通道归一化层。
    Fixed channel normalization layer.

    使用 pca_priors_full.pt 中保存的 norm_center / norm_scale，不使用在线 batch 统计。
    English: norm_scale，不使用在线 batch 统计.
    """

    def __init__(self, center: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6):
        """
        初始化固定逐通道 z-score 归一化参数。
        English: z-score parameter.

        输入:
        English: Input:
            center: [8] 离线统计均值。
            English: center: [8] .
            scale: [8] 离线统计标准差。
            English: scale: [8] .
            eps: 数值稳定项。
            English: eps: .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        center = torch.as_tensor(center, dtype=torch.float32).flatten()
        scale = torch.as_tensor(scale, dtype=torch.float32).flatten()

        if center.numel() != IMAGE_CHANNELS:
            raise ValueError(f"norm_center 应为 {IMAGE_CHANNELS} 维，当前为 {center.numel()}。")
        if scale.numel() != IMAGE_CHANNELS:
            raise ValueError(f"norm_scale 应为 {IMAGE_CHANNELS} 维，当前为 {scale.numel()}。")

        self.register_buffer("center", center.view(1, -1, 1, 1))
        self.register_buffer("scale", torch.clamp(scale, min=float(eps)).view(1, -1, 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入频层特征执行固定归一化。
        English: Input.

        输入:
        English: Input:
            x: [B, 8, H, W] 频层特征图。
            English: x: [B, 8, H, W] .
        输出:
        English: Output:
            与 x 同尺寸的归一化结果。
            English: x result.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return (x - self.center) / (self.scale + self.eps)


class FixedPCAProjection(nn.Module):
    """
    固定 PCA 1×1 投影层。
    Fixed PCA 1×1 projection layer.

    权重来自离线 PCA 先验，默认冻结。
    English: PCA , default.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, freeze: bool = True):
        """
        初始化固定 PCA 1x1 卷积投影。
        English: PCA 1x1 .

        输入:
        English: Input:
            weight: [K, 8] PCA 主成分方向。
            English: weight: [K, 8] PCA .
            bias: [K] PCA 均值折算偏置。
            English: bias: [K] PCA .
            freeze: True 时不参与反向传播更新。
            English: freeze: True update.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        weight = torch.as_tensor(weight, dtype=torch.float32)
        bias = torch.as_tensor(bias, dtype=torch.float32).flatten()

        if weight.dim() != 2:
            raise ValueError(f"PCA weight 必须为 [K, C]，当前 shape={tuple(weight.shape)}。")

        out_channels, in_channels = weight.shape
        if in_channels != IMAGE_CHANNELS:
            raise ValueError(f"PCA weight 输入维度必须为 {IMAGE_CHANNELS}，当前为 {in_channels}。")
        if bias.numel() != out_channels:
            raise ValueError(f"PCA bias 长度必须为 {out_channels}，当前为 {bias.numel()}。")

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        with torch.no_grad():
            self.proj.weight.copy_(weight.view(out_channels, in_channels, 1, 1))
            self.proj.bias.copy_(bias)

        if freeze:
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 8 通道频层特征投影到 K 个固定 PCA 主成分图。
        English: 8 K PCA .

        输入:
            x: [B, 8, H, W]。
        输出:
            [B, K, H, W]。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return self.proj(x)


class BandPriorProjection(nn.Module):
    """
    频层固定先验投影模块。
    Band-wise fixed prior projection.

    Pipeline:
        raw pyramid band -> fixed channel normalization -> fixed PCA 1×1 projection

    额外保存：
        effective_feature_vector_ratio:
            由离线先验构建阶段统计得到的频层有效特征向量占比 N_l，
            English: build N_l,.
            用于 PCASE 决定尺度压缩后的总通道容量。
            English: PCASE .
    """

    def __init__(self, band_prior: Dict, band_name: str, freeze_pca: bool = True):
        """
        根据单个频层 prior 构造“固定归一化 + 固定 PCA”模块。
        English: prior “ + PCA”.

        输入:
        English: Input:
            band_prior: pca_priors_full.pt 中某个频层的先验字典。
            band_name: high1/high2/high3/low。
            freeze_pca: 是否冻结 PCA 1x1 卷积参数。
            English: freeze_pca: PCA 1x1 parameter.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        center = _get_first_available(band_prior, ["norm_center", "center"], band_name)
        scale = _get_first_available(band_prior, ["norm_scale", "scale"], band_name)
        weight = _get_first_available(band_prior, ["weight", "pca_weight"], band_name)
        bias = _get_first_available(band_prior, ["bias", "pca_bias"], band_name)
        eigvals = _get_first_available(band_prior, ["eigvals", "eigenvalues"], band_name)
        evr = _get_first_available(
            band_prior,
            ["explained_variance_ratio", "explained_ratio", "evr"],
            band_name,
        )

        self.band_name = str(band_name)

        self.normalizer = FixedChannelNormalizer(
            center=center,
            scale=scale,
            eps=float(band_prior.get("norm_eps", 1e-6)),
        )
        self.pca = FixedPCAProjection(weight=weight, bias=bias, freeze=freeze_pca)

        eigvals = torch.as_tensor(eigvals, dtype=torch.float32).flatten()
        evr = torch.as_tensor(evr, dtype=torch.float32).flatten()

        if eigvals.numel() != self.pca.proj.out_channels:
            raise ValueError(f"{band_name}: eigvals 数量与 PCA 输出通道数不一致。")
        if evr.numel() != self.pca.proj.out_channels:
            raise ValueError(f"{band_name}: explained_variance_ratio 数量与 PCA 输出通道数不一致。")

        self.register_buffer("eigvals", eigvals, persistent=False)
        self.register_buffer("explained_variance_ratio", evr, persistent=False)

        # ================= PCASE 容量先验读取 =================
        # EN: ================= PCASE amount first read =================.
        # 逻辑：
        # EN: Logic:
        # 1. 新版 pca_priors_full.pt 应包含 effective_feature_vector_ratio；
        # EN: new version pca_priors_full.pt should effective_feature_vector_ratio;
        # 2. 若旧版先验没有该字段，则返回 None；
        # EN: if old version first not this field, then return None;
        # 3. 后续 PCASE 会在 prior_feature_vector_ratio=None 时退回到
        # EN: later PCASE will in prior_feature_vector_ratio=None when to.
        #    input_feature_vector_ratio，保证旧先验仍可运行。
        # EN: input_feature_vector_ratio, ensure old first still can run.
        self.effective_feature_vector_ratio = _get_optional_prior_float(
            band_prior,
            [
                "effective_feature_vector_ratio",
                "feature_vector_ratio",
                "structural_kept_ratio",
                "kept_ratio",
            ],
            default=None,
        )

        self.total_candidate_vectors = _get_optional_prior_float(
            band_prior,
            ["total_candidate_vectors", "candidate_vector_count", "vector_count"],
            default=None,
        )
        self.effective_vector_count = _get_optional_prior_float(
            band_prior,
            ["effective_vector_count", "kept_count", "merged_vector_count"],
            default=None,
        )

    @property
    def num_components(self) -> int:
        """
        返回当前频层保留的 PCA 主成分数量 K_l。
        English: returncurrent PCA K_l.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return int(self.eigvals.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行频层固定先验投影。
        English: This docstring documents the corresponding function behavior and engineering constraints.

        输入:
        English: Input:
            x: [B, 8, H, W] 原始拉普拉斯频层。
            English: x: [B, 8, H, W] .
        输出:
        English: Output:
            [B, K_l, H, W] PCA 主成分特征图。
            English: [B, K_l, H, W] PCA .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return self.pca(self.normalizer(x))


# =============================================================================
# 3. Allocation and lightweight downsampling
# =============================================================================

def _ceil_int(x: float) -> int:
    """
    对浮点数执行向上取整并返回 int。
    English: return int.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    return int(math.ceil(float(x)))


def _safe_log2_int(scale: int) -> int:
    """
    校验空间压缩倍率是否为 2 的整数次幂，并返回 log2(scale)。
    English: validation 2 , return log2(scale).

    输入:
    English: Input:
        scale: 空间边长压缩倍率。
        English: scale: .
    输出:
    English: Output:
        整数下采样次数。
        English: This docstring documents the corresponding function behavior and engineering constraints.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    scale = int(scale)
    n = int(round(math.log2(scale)))
    if 2 ** n != scale:
        raise ValueError(f"空间压缩倍率必须为 2 的整数次幂，当前 scale={scale}。")
    return n


def calculate_effective_pca_rank(eigvals: torch.Tensor, eps: float = 1e-12) -> float:
    """
    计算由 PCA 特征值分布决定的有效主成分秩。
    Calculate the effective PCA rank from the eigenvalue distribution.

    逻辑：
    English: Logic:
    1. K_l 只表示累计解释方差阈值下保留了多少个 PC 分支；
    English: 1. K_l PC ;
    2. 若直接用 K_l 放大 PCASE 总容量，低贡献 PC 也会线性抬高通道数；
    English: 2. K_l PCASE , PC ;
    3. 因此使用 participation-ratio 形式的有效秩：
       K_eff = (sum(lambda))^2 / sum(lambda^2)；
    4. 当各 PC 贡献接近时 K_eff 接近 K_l；当少数 PC 主导时 K_eff 接近实际主导维数。
    English: 4. PC K_eff K_l; PC K_eff .
    """
    eigvals = torch.as_tensor(eigvals, dtype=torch.float32).flatten().clamp_min(0)
    if eigvals.numel() <= 0:
        raise ValueError("eigvals 不能为空，无法计算 effective PCA rank。")

    eig_sum = float(eigvals.sum().item())
    eig_sq_sum = float(torch.sum(eigvals * eigvals).item())
    if eig_sum <= 0.0 or eig_sq_sum <= 0.0:
        return 1.0

    effective_rank = (eig_sum * eig_sum) / max(eig_sq_sum, float(eps))
    return float(max(1.0, min(effective_rank, float(eigvals.numel()))))


def allocate_pc_channels(weights: torch.Tensor, total_channels: int, min_dim: int = 1) -> List[int]:
    """
    按 PCA 权重分配每个 PC 分支的输出通道。
    Allocate output channels for PC branches.

    规则：
    English: :
    1. 每个 PC 至少 min_dim 个通道；
    English: 1. PC min_dim ;
    2. 剩余通道按 eigvals 或 explained_variance_ratio 比例分配；
    English: 2. eigvals explained_variance_ratio ;
    3. 最大余数法补齐，严格保证 sum(alloc) == total_channels。
    English: 3. , ensure sum(alloc) == total_channels.
    """
    weights = torch.as_tensor(weights, dtype=torch.float32).flatten().clamp_min(0)
    num_pc = int(weights.numel())
    total_channels = int(total_channels)
    min_dim = int(min_dim)

    if num_pc <= 0:
        raise ValueError("weights 不能为空。")
    if total_channels < num_pc * min_dim:
        raise ValueError(f"total_channels={total_channels} 小于 num_pc×min_dim={num_pc * min_dim}。")

    if float(weights.sum().item()) <= 0:
        weights = torch.ones_like(weights) / float(num_pc)
    else:
        weights = weights / weights.sum()

    alloc = torch.full((num_pc,), min_dim, dtype=torch.int64)
    remaining = total_channels - int(alloc.sum().item())

    if remaining <= 0:
        return [int(x) for x in alloc.tolist()]

    ideal_extra = remaining * weights
    floor_extra = torch.floor(ideal_extra).to(torch.int64)
    alloc = alloc + floor_extra

    leftover = total_channels - int(alloc.sum().item())
    if leftover > 0:
        remainders = (ideal_extra - floor_extra.to(ideal_extra.dtype)).tolist()
        order = sorted(range(num_pc), key=lambda i: remainders[i], reverse=True)
        for idx in range(leftover):
            alloc[order[idx % num_pc]] += 1

    if int(alloc.sum().item()) != total_channels:
        raise RuntimeError("PC 通道分配失败。")

    return [int(x) for x in alloc.tolist()]


def build_branch_channel_schedule(final_dim: int, num_downsamples: int) -> List[int]:
    """
    根据最终通道数构建逐级通道表。
    Build progressive channels by the n-th-root rule.
    """
    final_dim = int(final_dim)
    num_downsamples = int(num_downsamples)

    if final_dim <= 0:
        raise ValueError("final_dim 必须为正整数。")
    if num_downsamples <= 0:
        raise ValueError("num_downsamples 必须为正整数。")

    channels = []
    for s in range(1, num_downsamples + 1):
        if s == num_downsamples:
            c = final_dim
        else:
            c = max(1, int(round(final_dim ** (s / num_downsamples))))
            c = min(c, final_dim)
        channels.append(c)

    for i in range(1, len(channels)):
        channels[i] = max(channels[i], channels[i - 1])

    channels[-1] = final_dim
    return channels


def make_channel_safe_norm2d(num_channels: int, max_groups: int = 8) -> nn.BatchNorm2d:
    """
    构建冻结 BN 训练策略使用的 2D 通道归一化层。
    Build the 2D normalization layer used by the frozen-BN training policy.

    设计原因：
    English: :
    1. MFPC-HFNet 在 H3 -> low 融合后会形成 [B, C, 1, 1] token；
    English: 1. MFPC-HFNet H3 -> low [B, C, 1, 1] token;
    2. 当前撤销 GroupNorm 结构，恢复 BatchNorm2d，保证模型结构与旧 BN 设计一致；
    English: 2. current GroupNorm , BatchNorm2d, ensuremodel BN ;
    3. 当训练 micro-batch 很小时，由训练脚本将 BatchNorm2d 置为 eval，
    English: 3. training micro-batch , training BatchNorm2d eval,.
       冻结 running statistics，避免 [B, C, 1, 1] 上的 batch 统计失效；
       English: running statistics, avoid [B, C, 1, 1] batch ;
    4. 最近修改时间：2026-05-22。
    English: 4. Last modified: 2026-05-22.
    """
    num_channels = int(num_channels)
    if num_channels <= 0:
        raise ValueError(f"num_channels 必须为正整数，当前为 {num_channels}。")
    return nn.BatchNorm2d(num_channels)


def _json_none_or_float(value):
    """
    将可选数值或 tensor 转为 JSON 可保存的 float。
    English: optional tensor JSON save float.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().flatten()[0].item()
    return float(value)


def _json_none_or_int(value):
    """
    将可选数值或 tensor 转为 JSON 可保存的 int。
    English: optional tensor JSON save int.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().flatten()[0].item()
    return int(round(float(value)))


def summarize_pcase_band(band_name: str, prior, pcase, enabled: bool = True) -> Dict:
    """
    汇总单个频层中所有与 PCA/PCASE 先验容量计算相关的关键参数。
    Summarize PCA/PCASE prior-derived capacity parameters for one frequency band.

    记录原则：
    English: :
    1. 同时记录 PCA 先验侧信息，例如主成分数、有效向量数、有效向量占比；
    English: 1. PCA , , , ;
    2. 同时记录 PCASE 容量公式中的中间量，例如空间压缩倍率、S²×N_l、ceil 前通道数；
    English: 2. PCASE , , S²×N_l, ceil ;
    3. 同时记录最终生效量，例如 total_out_channels 和每个 PC 分支的 branch_dims；
    English: 3. , total_out_channels PC branch_dims;
    4. 对消融中未启用的频层，enabled=False，并显式标记 not_instantiated。
    English: 4. , enabled=False, explicit not_instantiated.
    """
    band_name = str(band_name)
    if (not enabled) or prior is None or pcase is None:
        return {
            "band_name": band_name,
            "enabled": False,
            "status": "not_instantiated_in_this_ablation_mode",
        }

    capacity = pcase.get_capacity_summary()
    capacity.update({
        "band_name": band_name,
        "enabled": True,
        "pca_num_components": int(prior.num_components),
        "pca_projection_out_channels": int(prior.num_components),
        "prior_effective_feature_vector_ratio": _json_none_or_float(prior.effective_feature_vector_ratio),
        "prior_total_candidate_vectors": _json_none_or_int(prior.total_candidate_vectors),
        "prior_effective_vector_count": _json_none_or_int(prior.effective_vector_count),
        "eigvals_sum": float(prior.eigvals.detach().cpu().sum().item()),
        "explained_variance_ratio_sum": float(prior.explained_variance_ratio.detach().cpu().sum().item()),
    })
    return capacity


class ConvBNAct(nn.Module):
    """Conv + BN + activation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, stride: int = 1, groups: int = 1):
        """
        构建 Conv2d + BatchNorm2d + SiLU 基础块。
        English: build Conv2d + BatchNorm2d + SiLU .

        输入:
        English: Input:
            in_ch / out_ch: 输入输出通道。
            English: out_ch: 输入输出通道.
            kernel_size / stride / groups: 卷积核、步长和分组参数。
            English: stride / groups: 卷积核、步长和分组参数.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行基础卷积块前向传播。
        English: This docstring documents the corresponding function behavior and engineering constraints.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return self.net(x)


# ================= EfficientNetV2-style PCASE compression =================
# 逻辑：
# EN: Logic:
# 1. V2 只替换 PCASE 内部每个 PC 分支的下采样压缩块；
# EN: V2 only PCASE inside each PC below;
# 2. PCASE 总容量、每个 PC 分支最终输出维度、后续 LD Encoder 与融合拓扑保持不变；
# EN: PCASE amount, each PC most degree, later LD Encoder and fusion not;
# 3. 这里按 224 -> 128 的复合缩放同步压缩深度和宽度，用作 128×128 输入口径的块级结构；
# EN: here by 224 -> 128 same deep degree and degree, use 128x128 interface result;
# 4. 最近修改时间：2026-05-13。
# EN: Last modified: 2026-05-13.
EFFICIENTNETV2_PCASE_BASE_RESOLUTION = 224
EFFICIENTNETV2_PCASE_TARGET_RESOLUTION = 128
EFFICIENTNETV2_PCASE_ALPHA = 1.2
EFFICIENTNETV2_PCASE_BETA = 1.1
EFFICIENTNETV2_PCASE_GAMMA = 1.15
EFFICIENTNETV2_PCASE_BASE_STAGE_CHANNELS = (24, 48, 64, 128)
EFFICIENTNETV2_PCASE_BASE_STAGE_REPEATS = (2, 4, 4, 6)
EFFICIENTNETV2_PCASE_STAGE_TYPES = ("fused", "fused", "fused", "mbconv_se")
EFFICIENTNETV2_PCASE_STAGE_EXPAND_RATIOS = (1, 4, 4, 4)


def get_pcase_efficientnetv2_128_scaling() -> Tuple[float, float, float]:
    """
    计算 PCASE 内部 128×128 EfficientNetV2-style 压缩结构的复合缩放系数。
    Calculate compound scaling factors for the 128×128 PCASE EfficientNetV2-style compressor.

    物理/算法含义：
    English: /:
    1. 以 224 作为 EfficientNet 系列常用基准分辨率；
    English: 1. 224 EfficientNet ;
    2. 通过 128 = 224 × gamma^phi 反推 phi；
    3. width_mult = beta^phi，depth_mult = alpha^phi；
    4. 最近修改时间：2026-05-13。
    English: 4. Last modified: 2026-05-13.
    """
    phi = math.log(EFFICIENTNETV2_PCASE_TARGET_RESOLUTION / EFFICIENTNETV2_PCASE_BASE_RESOLUTION) / math.log(
        EFFICIENTNETV2_PCASE_GAMMA
    )
    width_mult = EFFICIENTNETV2_PCASE_BETA ** phi
    depth_mult = EFFICIENTNETV2_PCASE_ALPHA ** phi
    return float(phi), float(width_mult), float(depth_mult)


def _make_divisible_channels(channels: float, divisor: int = 8) -> int:
    """
    按 EfficientNet 常用 8 通道粒度对齐 stage 宽度。
    English: EfficientNet 8 stage .
    """
    channels = float(channels)
    divisor = int(divisor)
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


def get_pcase_efficientnetv2_128_stage_summary() -> Dict:
    """
    返回 V2 PCASE 压缩结构的 stage 深度/宽度摘要。
    Return the stage depth/width summary for the V2 PCASE compressor.
    """
    phi, width_mult, depth_mult = get_pcase_efficientnetv2_128_scaling()
    stage_channels = [
        _make_divisible_channels(base_ch * width_mult)
        for base_ch in EFFICIENTNETV2_PCASE_BASE_STAGE_CHANNELS
    ]
    stage_repeats = [
        max(1, int(math.ceil(base_repeat * depth_mult)))
        for base_repeat in EFFICIENTNETV2_PCASE_BASE_STAGE_REPEATS
    ]
    return {
        "compression_block": "EfficientNetV2-style 128x128 compound-scaled PCASE branch",
        "paper_model_name_note": "Code name uses V2, paper name remains MFPC-HFNet.",
        "base_resolution": int(EFFICIENTNETV2_PCASE_BASE_RESOLUTION),
        "target_resolution": int(EFFICIENTNETV2_PCASE_TARGET_RESOLUTION),
        "phi": float(phi),
        "width_mult": float(width_mult),
        "depth_mult": float(depth_mult),
        "stage_types": list(EFFICIENTNETV2_PCASE_STAGE_TYPES),
        "stage_expand_ratios": [int(x) for x in EFFICIENTNETV2_PCASE_STAGE_EXPAND_RATIOS],
        "base_stage_channels": [int(x) for x in EFFICIENTNETV2_PCASE_BASE_STAGE_CHANNELS],
        "scaled_stage_channels": [int(x) for x in stage_channels],
        "base_stage_repeats": [int(x) for x in EFFICIENTNETV2_PCASE_BASE_STAGE_REPEATS],
        "scaled_stage_repeats": [int(x) for x in stage_repeats],
        "stage_strides": [2, 2, 2, 2],
        "recent_modified_at": "2026-05-13",
    }


class SqueezeExcitation2d(nn.Module):
    """
    EfficientNet MBConv 使用的轻量通道注意力。
    English: EfficientNet MBConv .
    """

    def __init__(self, in_ch: int, squeeze_ch: int):
        """
        初始化 SE 通道重标定模块。
        English: SE .

        输入:
        English: Input:
            in_ch: 输入通道数。
            English: in_ch: Input.
            squeeze_ch: 全局池化后瓶颈通道数。
            English: squeeze_ch: .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, squeeze_ch, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_ch, in_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输出通道注意力加权后的特征图。
        English: Output.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return x * self.net(x)


class EfficientNetV2FusedBlock(nn.Module):
    """
    EfficientNetV2 Fused-MBConv 风格块。
    English: EfficientNetV2 Fused-MBConv .

    修改说明：
    English: :
    1. 用于 PCASE 分支内 128×128 口径压缩；
    English: 1. PCASE 128×128 ;
    2. stride=2 时完成空间下采样，stride=1 且通道一致时保留残差；
    English: 2. stride=2 , stride=1 ;
    3. 最近修改时间：2026-05-13。
    English: 3. Last modified: 2026-05-13.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        """
        初始化 Fused-MBConv 风格卷积块。
        English: Fused-MBConv .

        输入:
        English: Input:
            in_ch / out_ch: 输入输出通道。
            English: out_ch: 输入输出通道.
            stride: 1 表示保留空间尺寸，2 表示下采样。
            English: stride: 1 , 2 .
            expand_ratio: 中间扩展通道倍率。
            English: expand_ratio: .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        self.use_residual = int(stride) == 1 and int(in_ch) == int(out_ch)
        expanded_ch = int(in_ch) * int(expand_ratio)

        if int(expand_ratio) == 1:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, expanded_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(expanded_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(expanded_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 Fused-MBConv 前向传播，并在尺寸/通道一致时叠加残差。
        English: Fused-MBConv , /.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        out = self.net(x)
        if self.use_residual:
            out = out + x
        return out


class EfficientNetV2MBConvSEBlock(nn.Module):
    """
    EfficientNetV2 MBConv+SE 风格块。
    English: EfficientNetV2 MBConv+SE .

    修改说明：
    English: :
    1. 用于 PCASE 分支的后段压缩，补充深度可分离卷积与 SE 通道重标定；
    English: 1. PCASE , SE ;
    2. 输出投影后只在 stride=1 且通道一致时使用残差；
    English: 2. Output stride=1 ;
    3. 最近修改时间：2026-05-13。
    English: 3. Last modified: 2026-05-13.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        """
        初始化 MBConv+SE 风格卷积块。
        English: MBConv+SE .

        输入:
        English: Input:
            in_ch / out_ch: 输入输出通道。
            English: out_ch: 输入输出通道.
            stride: stage 首块通常为 2，用于下采样。
            English: stride: stage 2, .
            expand_ratio: 倒残差扩展倍率。
            English: expand_ratio: .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        self.use_residual = int(stride) == 1 and int(in_ch) == int(out_ch)
        expanded_ch = int(in_ch) * int(expand_ratio)
        squeeze_ch = max(1, int(in_ch) // 4)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, expanded_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(expanded_ch, expanded_ch, kernel_size=3, stride=stride, padding=1, groups=expanded_ch, bias=False),
            nn.BatchNorm2d(expanded_ch),
            nn.SiLU(inplace=True),
            SqueezeExcitation2d(expanded_ch, squeeze_ch),
            nn.Conv2d(expanded_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 MBConv+SE 前向传播，并在允许时使用残差连接。
        English: MBConv+SE , .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        out = self.net(x)
        if self.use_residual:
            out = out + x
        return out


class LightweightDownsampleBlock(nn.Module):
    """
    轻量下采样残差块。
    Lightweight residual downsampling block.

    Main path:
        depthwise 3×3 stride=2 -> pointwise 1×1
    Shortcut:
        avgpool stride=2 -> optional pointwise projection
    """

    def __init__(self, in_ch: int, out_ch: int):
        """
        初始化轻量下采样残差块。
        English: This docstring documents the corresponding function behavior and engineering constraints.

        输入:
        English: Input:
            in_ch / out_ch: 输入输出通道数。
            English: out_ch: 输入输出通道数.
        说明:
        English: :
            主路径用 depthwise + pointwise，下采样捷径用 avgpool，保证空间尺寸对齐。
            English: path depthwise + pointwise, avgpool, ensure.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.dw = ConvBNAct(in_ch, in_ch, kernel_size=3, stride=2, groups=in_ch)
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.shortcut_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if in_ch != out_ch:
            self.shortcut_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut_proj = nn.Identity()

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入特征下采样 2 倍并融合残差信息。
        English: Input 2 .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        identity = self.shortcut_proj(self.shortcut_pool(x))
        out = self.pw(self.dw(x))
        return self.act(out + identity)


class PCASEBranch(nn.Module):
    """
    单个 PC 分支自适应尺度嵌入分支。
    PC-wise adaptive scale embedding branch.

    V2 修改说明：
    English: V2 :
        1. 原版本使用 LightweightDownsampleBlock 串联完成 16 倍空间压缩；
        English: 1. LightweightDownsampleBlock 16 ;
        2. V2 改为 128×128 口径的 EfficientNetV2-style 压缩结构；
        English: 2. V2 128×128 EfficientNetV2-style ;
        3. 分支最终输出维度 final_dim 不变，保证 PCASE 总容量与后续 LD Encoder 接口不变；
        English: 3. Output final_dim , ensure PCASE LD Encoder ;
        4. 最近修改时间：2026-05-13。

    Input:
        [B, 1, H, W]
    Output:
        [B, d_i, H_out, W_out]
    """

    def __init__(self, final_dim: int, num_downsamples: int):
        """
        初始化单个 PCA 主成分对应的 PCASE 压缩分支。
        English: PCA PCASE .

        输入:
        English: Input:
            final_dim: 该 PC 分支最终分配的输出通道数。
            English: final_dim: PC Output.
            num_downsamples: 空间下采样次数；当前 128->8 固定为 4。
            English: num_downsamples: ; current 128->8 4.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        self.final_dim = int(final_dim)
        self.num_downsamples = int(num_downsamples)
        if self.num_downsamples != 4:
            raise ValueError(
                "SOC_MFPCHFNetV2 的 PCASEBranch 当前按 128×128 -> 8×8 口径设计，"
                f"要求 num_downsamples=4，当前为 {self.num_downsamples}。"
            )

        self.compression_summary = get_pcase_efficientnetv2_128_stage_summary()
        self.channel_schedule = [int(x) for x in self.compression_summary["scaled_stage_channels"]]
        self.stage_repeats = [int(x) for x in self.compression_summary["scaled_stage_repeats"]]
        self.stage_types = [str(x) for x in self.compression_summary["stage_types"]]
        self.stage_expand_ratios = [int(x) for x in self.compression_summary["stage_expand_ratios"]]

        # 说明：
        # EN: Notes:
        # 1. 每个 stage 的首个 block 使用 stride=2 完成下采样；
        # EN: each stage block use stride=2 complete below;
        # 2. 同一 stage 的后续重复 block 使用 stride=1 进行局部表征增强；
        # EN: same stage later block use stride=1 line table;
        # 3. 最后用 1×1 投影回 final_dim，使 PCASE 分配的输出通道数保持不变。
        # EN: most after use 1x1 final_dim, make PCASE pass number not.
        blocks = []
        in_ch = 1
        for stage_type, out_ch, repeats, expand_ratio in zip(
            self.stage_types, self.channel_schedule, self.stage_repeats, self.stage_expand_ratios
        ):
            for repeat_idx in range(int(repeats)):
                stride = 2 if repeat_idx == 0 else 1
                if stage_type == "fused":
                    blocks.append(EfficientNetV2FusedBlock(in_ch, out_ch, stride=stride, expand_ratio=expand_ratio))
                elif stage_type == "mbconv_se":
                    blocks.append(EfficientNetV2MBConvSEBlock(in_ch, out_ch, stride=stride, expand_ratio=expand_ratio))
                else:
                    raise ValueError(f"不支持的 EfficientNetV2 PCASE stage_type={stage_type}。")
                in_ch = out_ch

        blocks.append(ConvBNAct(in_ch, self.final_dim, kernel_size=1, stride=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将单个 PC map 压缩为目标 source map 分支特征。
        English: PC map source map .

        输入:
            x: [B, 1, H, W]。
        输出:
            [B, final_dim, H_out, W_out]。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return self.blocks(x)


class PrincipalComponentAdaptiveScaleEmbedding(nn.Module):
    """
    PCASE: Principal Component Adaptive Scale Embedding module.
    主成分自适应尺度嵌入模块。
    English: This docstring documents the corresponding function behavior and engineering constraints.

    容量计算逻辑：
        M_l = max(K_l × m_min, ceil(K_eff × S_l² × N_l))  # area_compensated
        M_l = max(K_l × m_min, ceil(K_eff × S_l × N_l))   # conv_like_channel_doubling

    其中：
    English: :
        K_l   : 当前频层 PCA 主成分数，决定实际 PC 分支数量；
        English: K_l : current PCA , PC ;
        K_eff : 由 PCA 特征值分布计算的有效主成分秩，用于抑制低贡献 PC 导致的容量膨胀；
        English: K_eff : PCA calculate, PC ;
        S_l   : 空间压缩倍率 input_size / target_size；
        English: target_size；.
        N_l   : 离线先验统计得到的有效特征向量占比；
        English: N_l : ;
        m_min : 每个主成分分支的最小嵌入维度。
        English: m_min : .

    兼容逻辑：
    English: compatibleLogic:
        若 prior_feature_vector_ratio 缺失，则退回 input_feature_vector_ratio。
        English: prior_feature_vector_ratio missing, input_feature_vector_ratio.
        因此旧版 pca_priors_full.pt 仍可运行，但新版正式实验应优先使用先验字段。
        English: pca_priors_full.pt , field.
    """

    def __init__(
        self,
        num_components: int,
        eigvals: torch.Tensor,
        input_size: int,
        target_size: int,
        input_feature_vector_ratio: float,
        min_dim: int = 1,
        allocation_source: str = "eigvals",
        explained_variance_ratio: Optional[torch.Tensor] = None,
        prior_feature_vector_ratio: Optional[float] = None,
        capacity_mode: str = "area_compensated",
    ):
        """
        初始化 PCASE 并完成频层容量分配。
        English: PCASE .

        输入:
        English: Input:
            num_components: PCA 主成分数量 K_l。
            English: num_components: PCA K_l.
            eigvals: PCA 特征值，用于 effective PCA rank 和默认通道分配。
            English: eigvals: PCA , effective PCA rank default.
            input_size / target_size: 当前频层进入/输出 PCASE 的空间边长。
            English: target_size: 当前频层进入/输出 PCASE 的空间边长.
            input_feature_vector_ratio: 旧先验缺字段时的有效向量占比兜底。
            English: input_feature_vector_ratio: field.
            min_dim: 每个 PC 分支最少通道数。
            English: min_dim: PC .
            allocation_source: eigvals 或 explained_variance_ratio。
            English: allocation_source: eigvals explained_variance_ratio.
            explained_variance_ratio: 用于 EVR 分配时的贡献度。
            English: explained_variance_ratio: EVR .
            prior_feature_vector_ratio: 新版先验记录的有效结构向量占比 N_l。
            English: prior_feature_vector_ratio: N_l.
            capacity_mode: 高频 area_compensated 或低频 conv_like_channel_doubling。
            English: capacity_mode: area_compensated conv_like_channel_doubling.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.num_components = int(num_components)
        self.input_size = int(input_size)
        self.target_size = int(target_size)
        self.fallback_input_feature_vector_ratio = float(input_feature_vector_ratio)
        self.min_dim = int(min_dim)

        if self.input_size % self.target_size != 0:
            raise ValueError(f"input_size={input_size} 必须能整除 target_size={target_size}。")

        self.scale = self.input_size // self.target_size
        self.scale_squared = int(self.scale ** 2)
        self.num_downsamples = _safe_log2_int(self.scale)

        if prior_feature_vector_ratio is not None:
            self.effective_feature_vector_ratio = float(prior_feature_vector_ratio)
            self.feature_vector_ratio_source = "prior_effective_feature_vector_ratio"
        else:
            self.effective_feature_vector_ratio = float(input_feature_vector_ratio)
            self.feature_vector_ratio_source = "fallback_input_feature_vector_ratio"

        if self.effective_feature_vector_ratio <= 0:
            raise ValueError(
                f"effective_feature_vector_ratio 必须为正数，当前为 {self.effective_feature_vector_ratio}。"
            )

        self.effective_feature_vector_ratio = min(float(self.effective_feature_vector_ratio), 1.0)

        # ================= PCASE 容量中间量显式保存 =================
        # EN: ================= PCASE amount in amount form save =================.
        # 逻辑：
        # EN: Logic:
        # 1. K_l 仅表示实际保留的 PC 分支数量，并用于 K_l × m_min 的分支保底；
        # EN: K_l onlymeans keep PC count, and use K_l x m_min;
        # 2. PCASE 总容量不再直接乘 K_l，而是乘由 eigvals 分布得到的 effective_pca_rank；
        # EN: PCASE amount no longerdirectly K_l, is by eigvals obtain effective_pca_rank;
        # 3. high1 / high2 / high3 仍使用 area_compensated，即 K_eff × S_l² × N_l；
        # EN: high1 / high2 / high3 still use area_compensated, K_eff x S_l² x N_l;
        # 4. low 频层按当前修正版使用 conv_like_channel_doubling，即 K_eff × S_l × N_l；
        # EN: low frequency band by current use conv_like_channel_doubling, K_eff x S_l x N_l;
        # 5. 这样 low 不再因为 N_l=1 而按面积压缩倍率膨胀到过多通道；
        # EN: low no longerbecause N_l=1 by compression ratio to more pass;
        # 6. total_out_channels 是最终进入 LD Encoder 前的 PCASE 输出通道数；
        # EN: total_out_channels is most LD Encoder before PCASE pass number;
        # 7. realized_capacity_multiplier 记录 ceil 和 min 约束后实际生效的容量倍率。
        # EN: realized_capacity_multiplier ceil and min after amount multiplier.
        self.capacity_mode = str(capacity_mode).lower().strip()
        self.effective_pca_rank = calculate_effective_pca_rank(eigvals)
        self.effective_pca_rank_source = "participation_ratio_from_eigvals"

        self.area_compensated_channel_scale_factor = float(self.scale_squared * self.effective_feature_vector_ratio)
        self.conv_like_channel_scale_factor = float(self.scale * self.effective_feature_vector_ratio)

        if self.capacity_mode in ("area_compensated", "area", "scale_squared", "s2"):
            self.raw_channel_scale_factor = self.area_compensated_channel_scale_factor
            self.channel_scale_factor_source = "effective_rank_times_scale_squared_times_nl"
        elif self.capacity_mode in ("conv_like_channel_doubling", "conv_like", "scale", "s"):
            self.raw_channel_scale_factor = self.conv_like_channel_scale_factor
            self.channel_scale_factor_source = "effective_rank_times_scale_times_nl"
        else:
            raise ValueError(
                "capacity_mode 仅支持 area_compensated 或 conv_like_channel_doubling，"
                f"当前为 {capacity_mode}。"
            )

        # 旧公式保留为 manifest 诊断字段：K_l × scale_factor。
        # EN: old form keep as manifest field: K_l x scale_factor.
        # 新公式为：K_eff × scale_factor，同时仍用 K_l × m_min 作为每个 PC 分支的保底总量。
        # EN: new form as: K_eff x scale_factor, same when still use K_l x m_min as each PC amount.
        self.raw_total_channels_old_num_components = float(self.num_components * self.raw_channel_scale_factor)
        self.raw_total_channels = float(self.effective_pca_rank * self.raw_channel_scale_factor)
        self.min_total_channels = int(self.num_components * self.min_dim)
        self.total_out_channels = max(self.min_total_channels, _ceil_int(self.raw_total_channels))
        self.realized_channel_scale_factor = float(self.total_out_channels / max(self.num_components, 1))
        self.realized_capacity_multiplier = float(self.total_out_channels / max(self.raw_channel_scale_factor, 1e-12))

        allocation_source = str(allocation_source).lower().strip()
        if allocation_source == "eigvals":
            weights = eigvals
        elif allocation_source in ("evr", "explained_variance_ratio"):
            if explained_variance_ratio is None:
                raise ValueError("allocation_source='evr' 时必须传入 explained_variance_ratio。")
            weights = explained_variance_ratio
        else:
            raise ValueError(f"不支持的 allocation_source: {allocation_source}")
        self.allocation_source = allocation_source

        self.branch_dims = allocate_pc_channels(weights, self.total_out_channels, self.min_dim)
        self.pcase_branch_compression_summary = get_pcase_efficientnetv2_128_stage_summary()
        self.branches = nn.ModuleList(
            [PCASEBranch(final_dim=d, num_downsamples=self.num_downsamples) for d in self.branch_dims]
        )

    def get_capacity_summary(self) -> Dict:
        """
        返回 PCASE 容量计算与分配结果。
        Return PCASE capacity calculation and branch allocation summary.
        """
        branch_dims = [int(x) for x in self.branch_dims]
        return {
            "num_components": int(self.num_components),
            "input_size": int(self.input_size),
            "target_size": int(self.target_size),
            "spatial_compression_scale": int(self.scale),
            "spatial_compression_scale_squared": int(self.scale_squared),
            "num_downsamples": int(self.num_downsamples),
            "fallback_input_feature_vector_ratio": float(self.fallback_input_feature_vector_ratio),
            "effective_feature_vector_ratio": float(self.effective_feature_vector_ratio),
            "feature_vector_ratio_source": str(self.feature_vector_ratio_source),
            "capacity_mode": str(self.capacity_mode),
            "channel_scale_factor_source": str(self.channel_scale_factor_source),
            "effective_pca_rank": float(self.effective_pca_rank),
            "effective_pca_rank_source": str(self.effective_pca_rank_source),
            "raw_channel_scale_factor_used": float(self.raw_channel_scale_factor),
            "raw_channel_scale_factor_s_times_nl": float(self.conv_like_channel_scale_factor),
            "raw_channel_scale_factor_s2_times_nl": float(self.area_compensated_channel_scale_factor),
            "raw_total_channels_before_ceil": float(self.raw_total_channels),
            "raw_total_channels_before_ceil_old_num_components": float(self.raw_total_channels_old_num_components),
            "min_dim_per_component": int(self.min_dim),
            "min_total_channels": int(self.min_total_channels),
            "total_out_channels": int(self.total_out_channels),
            "realized_channel_scale_factor": float(self.realized_channel_scale_factor),
            "realized_capacity_multiplier": float(self.realized_capacity_multiplier),
            "allocation_source": str(self.allocation_source),
            "pcase_branch_compression": dict(self.pcase_branch_compression_summary),
            "branch_dims": branch_dims,
            "branch_dim_sum": int(sum(branch_dims)),
            "branch_dim_min": int(min(branch_dims)) if branch_dims else 0,
            "branch_dim_max": int(max(branch_dims)) if branch_dims else 0,
        }

    def forward(self, pc_maps: torch.Tensor) -> torch.Tensor:
        """
        对每个 PCA 主成分图分别执行 PCASE 分支压缩后拼接。
        English: PCA PCASE .

        输入:
            pc_maps: [B, K_l, H, W]。
        输出:
            [B, total_out_channels, target_size, target_size]。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if pc_maps.dim() != 4:
            raise ValueError(f"pc_maps 必须为 [B, K, H, W]，当前 shape={tuple(pc_maps.shape)}。")
        if pc_maps.shape[1] != self.num_components:
            raise ValueError(f"PC 通道数不一致：期望 {self.num_components}，实际 {pc_maps.shape[1]}。")

        outs = []
        for pc_idx, branch in enumerate(self.branches):
            outs.append(branch(pc_maps[:, pc_idx:pc_idx + 1]))
        return torch.cat(outs, dim=1)


# Backward-compatible alias for older experiment scripts.
PCScaleStem = PrincipalComponentAdaptiveScaleEmbedding
PCASE = PrincipalComponentAdaptiveScaleEmbedding


# ================= token 宽度自适应推导函数 =================
# EN: ================= token degree should function =================.
# 逻辑：
# EN: Logic:
# 1. PCASE 输出的是某一频层的 source map，真正进入 LD Encoder 的单个局部块为
# EN: PCASE is frequency band source map, actually LD Encoder single local block as.
#    patch_size × patch_size × pcase_channels；
# 2. token_compression_ratio 控制该局部块被压缩成 token 时的统一信息瓶颈；
# EN: token_compression_ratio control this local block token when information bottleneck;
# 3. 取整时使用向上取整到 round_multiple 的策略，避免实际压缩倍率超过目标倍率；
# EN: when use on to round_multiple, avoid compression ratio multiplier;
# 4. token_dim_min 用于保护 high2 / high3 等低容量频层，避免注意力宽度过窄。
# EN: token_dim_min use high2 / high3 low amount frequency band, avoid degree.
def ceil_to_multiple(value: float, multiple: int) -> int:
    """
    将数值向上取整到指定倍数。
    English: This docstring documents the corresponding function behavior and engineering constraints.

    输入:
    English: Input:
        value: 待取整数值。
        English: value: .
        multiple: 对齐倍数；小于等于 1 时退化为普通 ceil。
        English: multiple: ; 1 ceil.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    multiple = int(multiple)
    if multiple <= 1:
        return int(math.ceil(float(value)))
    return int(math.ceil(float(value) / float(multiple)) * multiple)


def derive_token_dim_from_patch_capacity(
    pcase_channels: int,
    patch_size: int,
    token_compression_ratio: float,
    token_dim_min: int = 96,
    token_dim_round_multiple: int = 16,
) -> int:
    """
    根据 PCASE source patch 的标量容量自适应推导 token 宽度。
    Derive token width from the scalar capacity of one PCASE source patch.

    Formula:
        source_flat_dim = patch_size * patch_size * pcase_channels
        token_dim = ceil_to_multiple(source_flat_dim / token_compression_ratio, token_dim_round_multiple)
        token_dim = max(token_dim, token_dim_min)
    """
    pcase_channels = int(pcase_channels)
    patch_size = int(patch_size)
    token_dim_min = int(token_dim_min)
    token_dim_round_multiple = int(token_dim_round_multiple)
    token_compression_ratio = float(token_compression_ratio)

    if pcase_channels <= 0:
        raise ValueError(f"pcase_channels 必须为正整数，当前为 {pcase_channels}。")
    if patch_size <= 0:
        raise ValueError(f"patch_size 必须为正整数，当前为 {patch_size}。")
    if token_compression_ratio <= 0:
        raise ValueError(f"token_compression_ratio 必须为正数，当前为 {token_compression_ratio}。")
    if token_dim_min <= 0:
        raise ValueError(f"token_dim_min 必须为正整数，当前为 {token_dim_min}。")
    if token_dim_round_multiple <= 0:
        raise ValueError(f"token_dim_round_multiple 必须为正整数，当前为 {token_dim_round_multiple}。")

    source_flat_dim = int(patch_size * patch_size * pcase_channels)
    raw_token_dim = float(source_flat_dim) / float(token_compression_ratio)
    rounded_token_dim = ceil_to_multiple(raw_token_dim, token_dim_round_multiple)
    return int(max(token_dim_min, rounded_token_dim))


def summarize_token_dim_derivation(
    pcase_channels: int,
    patch_size: int,
    token_dim: int,
    token_compression_ratio: float,
) -> Dict[str, Union[int, float]]:
    """
    记录 token 宽度自适应推导的中间量。
    English: token .

    输入:
    English: Input:
        pcase_channels: PCASE source map 通道数。
        English: pcase_channels: PCASE source map .
        patch_size: LD Encoder 的局部块尺寸。
        English: patch_size: LD Encoder .
        token_dim: 最终解析得到的 token 宽度。
        English: token_dim: parse token .
        token_compression_ratio: 目标压缩倍率。
        English: token_compression_ratio: .
    输出:
    English: Output:
        可写入 manifest 的推导摘要字典。
        English: write manifest dictionary.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    source_flat_dim = int(int(patch_size) * int(patch_size) * int(pcase_channels))
    token_dim = int(token_dim)
    return {
        "pcase_channels": int(pcase_channels),
        "patch_size": int(patch_size),
        "source_flat_dim": int(source_flat_dim),
        "target_token_compression_ratio": float(token_compression_ratio),
        "raw_token_dim_before_round": float(source_flat_dim) / float(token_compression_ratio),
        "resolved_token_dim": int(token_dim),
        "realized_token_compression_ratio": float(source_flat_dim) / float(token_dim),
    }


# =============================================================================
# 4. Patch splitting and LD Encoder
# =============================================================================

def split_to_nonoverlap_patches(x: torch.Tensor, patch_size: int = 8) -> Tuple[torch.Tensor, int, int]:
    """Split [B, C, H, W] into [B×Gh×Gw, C, patch_size, patch_size]."""
    if x.dim() != 4:
        raise ValueError(f"x 必须为 [B, C, H, W]，当前 shape={tuple(x.shape)}。")

    b, c, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"H/W 必须能被 patch_size={patch_size} 整除，当前 H={h}, W={w}。")

    gh, gw = h // patch_size, w // patch_size
    x = x.view(b, c, gh, patch_size, gw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(b * gh * gw, c, patch_size, patch_size), gh, gw


def restore_patch_tokens_to_grid(tokens: torch.Tensor, batch_size: int, gh: int, gw: int) -> torch.Tensor:
    """Restore [B×Gh×Gw, D] to [B, D, Gh, Gw]."""
    d = tokens.shape[1]
    grid = tokens.view(batch_size, gh, gw, d)
    return grid.permute(0, 3, 1, 2).contiguous()


class SimpleTokenSelfAttention(nn.Module):
    """
    低秩 token self-attention。
    Low-rank token self-attention.

    Input / output:
        [B, N, C]
    """

    def __init__(self, dim: int, attn_dim: Optional[int] = None, num_heads: int = 2, dropout: float = 0.0):
        """
        初始化低秩 self-attention。
        English: self-attention.

        输入:
        English: Input:
            dim: 输入 token 宽度。
            English: dim: Input token .
            attn_dim: QKV 内部注意力宽度；None 时按 dim 自动压缩。
            English: attn_dim: QKV ; None dim .
            num_heads: 注意力头数，会自动调整到能整除 attn_dim。
            English: num_heads: , attn_dim.
            dropout: 注意力权重 dropout。
            English: dropout: dropout.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        self.dim = int(dim)

        if attn_dim is None:
            attn_dim = min(64, max(16, self.dim // 2))

        attn_dim = int(attn_dim)
        num_heads = int(num_heads)

        if attn_dim < num_heads:
            num_heads = 1
        while num_heads > 1 and attn_dim % num_heads != 0:
            num_heads -= 1

        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = self.attn_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.dim, self.attn_dim * 3, bias=False)
        self.proj = nn.Linear(self.attn_dim, self.dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        对同一局部块或同一网格内的 token 执行 self-attention。
        English: token self-attention.

        输入:
            tokens: [B, N, C]。
        输出:
            [B, N, C]。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        b, n, _ = tokens.shape

        qkv = self.qkv(tokens)
        qkv = qkv.view(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.attn_dim)
        return self.proj(out)


class GatedConvFFN(nn.Module):
    """
    轻量 gated convolutional FFN。
    English: gated convolutional FFN.

    修改说明：
    English: :
    1. 原版本使用 BatchNorm2d；
    English: 1. BatchNorm2d;
    2. 当 CrossPatchEncoder 处理 [B, C, 1, 1] 且最后一个 batch 的 B=1 时，BatchNorm2d 会报错；
    English: 2. CrossPatchEncoder [B, C, 1, 1] batch B=1 , BatchNorm2d ;
    3. 当前恢复 BatchNorm2d，并由训练脚本在小 batch 时冻结 BN running statistics。
    English: 3. current BatchNorm2d, training batch BN running statistics.
    4. 最近修改时间：2026-05-22。
    English: 4. Last modified: 2026-05-22.
    """

    def __init__(self, dim: int, ffn_ratio: float = 1.5):
        """
        初始化门控卷积前馈网络。
        English: This docstring documents the corresponding function behavior and engineering constraints.

        输入:
        English: Input:
            dim: 输入输出通道。
            English: dim: InputOutput.
            ffn_ratio: 隐藏通道相对 dim 的倍率。
            English: ffn_ratio: dim .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        dim = int(dim)
        hidden = max(dim, int(round(dim * float(ffn_ratio))))

        self.expand = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=False)
        self.bn = make_channel_safe_norm2d(hidden * 2)
        self.project = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.project_bn = make_channel_safe_norm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行空间卷积增强和门控特征选择。
        English: select.

        输入:
            x: [B, C, H, W]。
        输出:
        English: Output:
            与 x 同形状的残差分支输出。
            English: x Output.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        x = self.bn(self.dw(self.expand(x)))
        a, gate = torch.chunk(x, 2, dim=1)
        x = a * torch.sigmoid(gate)
        return self.project_bn(self.project(x))


class LDEncoder(nn.Module):
    """
    LD Encoder: Local-Dynamic Encoder.

    输入一个 8×8 局部块，输出一个 block token。
    Encode one local 8×8 block into one token.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        attn_dim: Optional[int] = 64,
        num_heads: int = 2,
        ffn_ratio: float = 1.5,
        dropout: float = 0.0,
    ):
        """
        初始化单个局部块 LD Encoder。
        English: LD Encoder.

        输入:
        English: Input:
            in_dim: 局部块输入通道。
            English: in_dim: Input.
            out_dim: 输出 token 宽度。
            English: out_dim: Output token .
            attn_dim / num_heads: 块内 self-attention 配置。
            English: num_heads: 块内 self-attention 配置.
            ffn_ratio / dropout: 前馈网络和注意力 dropout 配置。
            English: dropout: 前馈网络和注意力 dropout 配置.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
        )
        self.pos_dw = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
        )

        self.norm1 = nn.LayerNorm(out_dim)
        self.attn = SimpleTokenSelfAttention(out_dim, attn_dim=attn_dim, num_heads=num_heads, dropout=dropout)
        self.ffn = GatedConvFFN(out_dim, ffn_ratio=ffn_ratio)
        self.pool_score = nn.Linear(out_dim, 1)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        将一个局部 patch 编码成一个 token。
        English: patch token.

        输入:
            patch: [B_patch, C, patch_size, patch_size]。
        输出:
            [B_patch, out_dim]。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        x = self.input_proj(patch)
        x = x + self.pos_dw(x)

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self.attn(self.norm1(tokens))

        x = tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        x = x + self.ffn(x)

        tokens = x.flatten(2).transpose(1, 2).contiguous()
        score = torch.softmax(self.pool_score(tokens), dim=1)
        return torch.sum(tokens * score, dim=1)


class PatchGridLDEncoder(nn.Module):
    """Source map -> 8×8 patches -> shared LD Encoder -> token grid."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: int = 8,
        attn_dim: Optional[int] = 64,
        num_heads: int = 2,
        ffn_ratio: float = 1.5,
        dropout: float = 0.0,
    ):
        """
        初始化 source map 到 token grid 的共享 LD 编码器。
        English: source map token grid LD .

        输入:
        English: Input:
            in_dim: PCASE source map 通道。
            English: in_dim: PCASE source map .
            out_dim: 每个 patch token 宽度。
            English: out_dim: patch token .
            patch_size: 非重叠局部块尺寸。
            English: patch_size: .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        self.patch_size = int(patch_size)
        self.encoder = LDEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            attn_dim=attn_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        将 source map 切成非重叠 patch 并恢复为 token grid。
        English: source map patch token grid.

        输入:
        English: Input:
            source: [B, C, H, W]，H/W 必须能被 patch_size 整除。
            English: source: [B, C, H, W], H/W patch_size .
        输出:
            [B, out_dim, H/patch_size, W/patch_size]。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        b = source.shape[0]
        patches, gh, gw = split_to_nonoverlap_patches(source, patch_size=self.patch_size)
        tokens = self.encoder(patches)
        return restore_patch_tokens_to_grid(tokens, batch_size=b, gh=gh, gw=gw)


class LowSelfLDEncoder(nn.Module):
    """
    Low self LD Encoder.

    当前 low 设计：
    English: current low :
    - 输入 low_source: [B, C_in, 16, 16]
    English: - Input low_source: [B, C_in, 16, 16]
    - 在 16×16 low patch 内做一次 self LD encoding；
    English: - 16×16 low patch self LD encoding;
    - 输出 low_patch: [B, D_out, 8, 8]
    English: - Output low_patch: [B, D_out, 8, 8]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        attn_dim: Optional[int] = 64,
        num_heads: int = 2,
        ffn_ratio: float = 1.5,
        dropout: float = 0.0,
    ):
        """
        初始化旧 low self-LD 编码器。
        English: low self-LD .

        输入:
        English: Input:
            in_dim / out_dim: low source 输入输出通道。
            English: out_dim: low source 输入输出通道.
            attn_dim / num_heads / ffn_ratio / dropout: self-attention 和 FFN 配置。
            English: num_heads / ffn_ratio / dropout: self-attention 和 FFN 配置.

        维护说明:
        English: :
            当前主路径使用 PatchGridLDEncoder 生成 1x1 low token；本类保留为结构兼容组件。
            English: currentpath PatchGridLDEncoder 1x1 low token; compatible.
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
        )
        self.pos_dw = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
        )

        self.norm1 = nn.LayerNorm(out_dim)
        self.attn = SimpleTokenSelfAttention(out_dim, attn_dim=attn_dim, num_heads=num_heads, dropout=dropout)
        self.ffn = GatedConvFFN(out_dim, ffn_ratio=ffn_ratio)

        # 一次 LD 编码后，将 16×16 压缩为 8×8 low patch。
        # EN: LD programming code after, 16x16 as 8x8 low patch.
        # After one self-LD encoding, compress 16×16 to an 8×8 low patch.
        self.to_low_patch = LightweightDownsampleBlock(out_dim, out_dim)

    def forward(self, low_source: torch.Tensor) -> torch.Tensor:
        """
        将 16x16 low source 编码并压缩为 8x8 low patch。
        English: 16x16 low source 8x8 low patch.

        输入:
            low_source: [B, C, 16, 16]。
        输出:
            [B, out_dim, 8, 8]。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if low_source.shape[-2:] != (16, 16):
            raise ValueError(f"LowSelfLDEncoder 期望输入为 16×16，当前为 {tuple(low_source.shape[-2:])}。")

        x = self.input_proj(low_source)
        x = x + self.pos_dw(x)

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self.attn(self.norm1(tokens))

        x = tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        x = x + self.ffn(x)

        low_patch = self.to_low_patch(x)

        if low_patch.shape[-2:] != (8, 8):
            raise RuntimeError(f"LowSelfLDEncoder 输出应为 8×8，当前为 {tuple(low_patch.shape[-2:])}。")

        return low_patch


# =============================================================================
# 5. Fusion, CrossPatchEncoder, and PACEBlock
# =============================================================================

class FusionBlock(nn.Module):
    """
    轻量融合块。
    Lightweight fusion block.

    Pipeline:
        1×1 Conv -> depthwise 3×3 Conv -> 1×1 Conv

    修改说明：
    English: :
    1. 该模块既用于 H1->H2、HF2->H3，也用于 H3->low；
    English: 1. H1->H2, HF2->H3, H3->low;
    2. H3->low 时输入空间尺寸为 1×1，若最后一个训练 batch 的 B=1，BatchNorm2d 会失效；
    English: 2. H3->low Input 1×1, training batch B=1, BatchNorm2d ;
    3. 当前恢复 BatchNorm2d，并依赖训练脚本的小 batch 冻结 BN 策略处理 1×1 token 路径。
    English: 3. current BatchNorm2d, training batch BN 1×1 token path.
    4. 最近修改时间：2026-05-22。
    English: 4. Last modified: 2026-05-22.
    """

    def __init__(self, in_dim: int, out_dim: int):
        """
        初始化高低频融合卷积块。
        English: This docstring documents the corresponding function behavior and engineering constraints.

        输入:
        English: Input:
            in_dim: 拼接后的输入通道。
            English: in_dim: Input.
            out_dim: 融合后的输出通道。
            English: out_dim: Output.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            make_channel_safe_norm2d(out_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False),
            make_channel_safe_norm2d(out_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            make_channel_safe_norm2d(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行局部卷积融合。
        English: This docstring documents the corresponding function behavior and engineering constraints.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return self.net(x)


class CrossPatchEncoder(nn.Module):
    """
    Cross Patch Encoder.
    跨块位置交互编码器。
    English: This docstring documents the corresponding function behavior and engineering constraints.

    用于跨频融合之后，让不同空间位置 / patch token 之间发生交互。
    English: patch token 之间发生交互.

    修改说明：
    English: :
    1. final_cpe 需要处理 [B, C, 1, 1] 的 low 对齐 token；
    English: 1. final_cpe [B, C, 1, 1] low token;
    2. pos_dw 中的归一化也必须避免 BatchNorm2d 的 batch-size 依赖；
    English: 2. pos_dw avoid BatchNorm2d batch-size ;
    3. 当前恢复 BatchNorm2d，训练阶段通过冻结 BN running statistics 避免小 batch 统计失效。
    English: 3. current BatchNorm2d, training BN running statistics avoid batch .
    4. 最近修改时间：2026-05-22。
    English: 4. Last modified: 2026-05-22.
    """

    def __init__(
        self,
        dim: int,
        attn_dim: Optional[int] = 64,
        num_heads: int = 2,
        ffn_ratio: float = 1.5,
        dropout: float = 0.0,
    ):
        """
        初始化跨 patch 位置交互编码器。
        English: patch .

        输入:
        English: Input:
            dim: 输入输出 token 通道。
            English: dim: InputOutput token .
            attn_dim / num_heads: 位置 token attention 配置。
            English: num_heads: 位置 token attention 配置.
            ffn_ratio / dropout: 卷积 FFN 和 dropout 配置。
            English: dropout: 卷积 FFN 和 dropout 配置.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.dim = int(dim)
        self.pos_dw = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim, bias=False),
            make_channel_safe_norm2d(self.dim),
            nn.SiLU(inplace=True),
        )
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = SimpleTokenSelfAttention(self.dim, attn_dim=attn_dim, num_heads=num_heads, dropout=dropout)
        self.ffn = GatedConvFFN(self.dim, ffn_ratio=ffn_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        在 [H, W] 网格上执行位置交互和卷积前馈增强。
        English: [H, W] .

        输入:
            x: [B, C, H, W]。
        输出:
            [B, C, H, W]。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        x = x + self.pos_dw(x)
        b, c, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self.attn(self.norm1(tokens))
        x = tokens.transpose(1, 2).contiguous().view(b, c, h, w)

        x = x + self.ffn(x)
        return x


class PACEBlock(nn.Module):
    """
    PACE Block:
    Position-Aligned Cross-Patch Expansion Block.

    中文：
    English: :
        位置对齐跨块扩展模块。
        English: This docstring documents the corresponding function behavior and engineering constraints.

    作用：
    English: :
        将父网格特征扩展到子网格，并在扩展后进行跨 patch 位置交互。

    Pipeline:
        1. 1×1 Conv generates child-position channel groups.
        2. PixelShuffle restores the child grid.
        3. CrossPatchEncoder enables cross-position interaction.

    Input:
        [B, C_in, H, W]

    Output:
        [B, C_out, 2H, 2W]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        upscale_factor: int = 2,
        cpe_attn_dim: Optional[int] = 64,
        cpe_heads: int = 2,
        cpe_ffn_ratio: float = 1.5,
        dropout: float = 0.0,
    ):
        """
        初始化位置对齐跨块扩展模块。
        English: This docstring documents the corresponding function behavior and engineering constraints.

        输入:
        English: Input:
            in_dim / out_dim: 输入输出通道。
            English: out_dim: 输入输出通道.
            upscale_factor: PixelShuffle 空间放大倍率。
            English: upscale_factor: PixelShuffle .
            cpe_*: 扩展后 CrossPatchEncoder 配置。
            English: cpe_*: CrossPatchEncoder configuration.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.upscale_factor = int(upscale_factor)

        if self.upscale_factor <= 1:
            raise ValueError("PACEBlock 的 upscale_factor 必须大于 1。")

        expanded_channels = self.out_dim * (self.upscale_factor ** 2)

        self.expand = nn.Sequential(
            nn.Conv2d(self.in_dim, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
        )
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        self.cross_patch = CrossPatchEncoder(
            dim=self.out_dim,
            attn_dim=cpe_attn_dim,
            num_heads=cpe_heads,
            ffn_ratio=cpe_ffn_ratio,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        先扩展父网格到子网格，再执行跨位置编码。
        English: , .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        x = self.expand(x)
        x = self.pixel_shuffle(x)
        x = self.cross_patch(x)
        return x


class HighFrequencySummaryAggregation(nn.Module):
    """
    HFSA:
    High-Frequency Summary Aggregation.

    H1 8×8 -> H2 4×4 -> H3 2×2 的单向高频摘要聚合。
    English: H1 8×8 -> H2 4×4 -> H3 2×2 .
    """

    def __init__(
        self,
        d1: int,
        d2: int,
        d3: int,
        dh2: int,
        dhf: int,
        cpe_attn_dim: Optional[int] = 64,
        cpe_heads: int = 2,
        cpe_ffn_ratio: float = 1.5,
        dropout: float = 0.0,
    ):
        """
        初始化高频摘要聚合模块。
        English: This docstring documents the corresponding function behavior and engineering constraints.

        输入:
        English: Input:
            d1/d2/d3: high1/high2/high3 token 宽度。
            English: d1/d2/d3: high1/high2/high3 token .
            dh2: H1 汇入 H2 后的中间摘要宽度。
            English: dh2: H1 H2 .
            dhf: H2 汇入 H3 后的高频摘要宽度。
            English: dhf: H2 H3 .
            cpe_*: 跨 patch 编码器配置。
            English: cpe_*: patch configuration.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.h1_to_h2 = nn.PixelUnshuffle(downscale_factor=2)
        self.fuse_h1_h2 = FusionBlock(4 * d1 + d2, dh2)
        self.cpe_hf2 = CrossPatchEncoder(dh2, cpe_attn_dim, cpe_heads, cpe_ffn_ratio, dropout)

        self.hf2_to_h3 = nn.PixelUnshuffle(downscale_factor=2)
        self.fuse_hf2_h3 = FusionBlock(4 * dh2 + d3, dhf)
        self.cpe_hf = CrossPatchEncoder(dhf, cpe_attn_dim, cpe_heads, cpe_ffn_ratio, dropout)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        按 H1 -> H2 -> H3 的父子路径聚合高频 token。
        English: H1 -> H2 -> H3 path token.

        输入:
            h1: [B, d1, 8, 8]。
            h2: [B, d2, 4, 4]。
            h3: [B, d3, 2, 2]。
        输出:
            hf_summary: [B, dhf, 2, 2]。
            aux: 中间高频摘要，供调试和可视化使用。
            English: aux: , .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        h1_parent = self.h1_to_h2(h1)
        hf2 = self.fuse_h1_h2(torch.cat([h1_parent, h2], dim=1))
        hf2 = self.cpe_hf2(hf2)

        hf2_parent = self.hf2_to_h3(hf2)
        hf_summary = self.fuse_hf2_h3(torch.cat([hf2_parent, h3], dim=1))
        hf_summary = self.cpe_hf(hf_summary)

        aux = {
            "h1_parent": h1_parent,
            "hf2": hf2,
            "hf2_parent": hf2_parent,
            "hf_summary": hf_summary,
        }
        return hf_summary, aux


class StructureAwareHighFrequencyAggregation(nn.Module):
    """
    结构消融高频摘要聚合模块。
    Structure-aware high-frequency aggregation for ablation variants.

    逻辑 / Logic:
    English: Logic:.
    1. Full 沿用 high1 -> high2 -> high3 的完整父子聚合；
    English: 1. Full high1 -> high2 -> high3 ;
    2. H2H3Low 从 high2 -> high3 开始聚合；
    English: 2. H2H3Low high2 -> high3 ;
    3. H3Low 直接把 high3 token 作为高频摘要；
    English: 3. H3Low high3 token ;
    4. LowOnly 不实例化本模块，图像分支直接使用 low token。
    English: 4. LowOnly , image low token.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    def __init__(
        self,
        active_high_bands: Sequence[str],
        d1: int,
        d2: int,
        d3: int,
        dh2: int,
        dhf: int,
        cpe_attn_dim: Optional[int] = 64,
        cpe_heads: int = 2,
        cpe_ffn_ratio: float = 1.5,
        dropout: float = 0.0,
    ):
        """
        初始化按结构裁剪后的高频聚合路径。
        English: path.

        输入:
        English: Input:
            active_high_bands: 当前启用的高频层，例如 high2/high3。
            English: active_high_bands: current, high2/high3.
            d1/d2/d3/dh2/dhf: 对应频层和摘要 token 宽度。
            English: d1/d2/d3/dh2/dhf: token .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.active_high_bands = tuple(active_high_bands)
        if self.active_high_bands == ("high1", "high2", "high3"):
            self.mode = "high1_high2_high3"
            self.full_hfsa = HighFrequencySummaryAggregation(
                d1=d1,
                d2=d2,
                d3=d3,
                dh2=dh2,
                dhf=dhf,
                cpe_attn_dim=cpe_attn_dim,
                cpe_heads=cpe_heads,
                cpe_ffn_ratio=cpe_ffn_ratio,
                dropout=dropout,
            )
        elif self.active_high_bands == ("high2", "high3"):
            self.mode = "high2_high3"
            self.h2_to_h3 = nn.PixelUnshuffle(downscale_factor=2)
            self.fuse_h2_h3 = FusionBlock(4 * d2 + d3, dhf)
            self.cpe_hf = CrossPatchEncoder(dhf, cpe_attn_dim, cpe_heads, cpe_ffn_ratio, dropout)
        elif self.active_high_bands == ("high3",):
            self.mode = "high3_only"
            self.h3_project = nn.Identity() if int(d3) == int(dhf) else nn.Conv2d(int(d3), int(dhf), kernel_size=1)
        else:
            raise ValueError(f"不支持的高频结构组合: {self.active_high_bands}。")

    def forward(
        self,
        h1: Optional[torch.Tensor],
        h2: Optional[torch.Tensor],
        h3: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        按启用结构生成 2x2 高频摘要。
        English: 2x2 .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if self.mode == "high1_high2_high3":
            if h1 is None or h2 is None or h3 is None:
                raise ValueError("Full 高频聚合需要 high1/high2/high3 token。")
            return self.full_hfsa(h1, h2, h3)

        if self.mode == "high2_high3":
            if h2 is None or h3 is None:
                raise ValueError("H2H3Low 高频聚合需要 high2/high3 token。")
            h2_parent = self.h2_to_h3(h2)
            hf_summary = self.fuse_h2_h3(torch.cat([h2_parent, h3], dim=1))
            hf_summary = self.cpe_hf(hf_summary)
            return hf_summary, {
                "h2_parent": h2_parent,
                "hf_summary": hf_summary,
            }

        if self.mode == "high3_only":
            if h3 is None:
                raise ValueError("H3Low 高频聚合需要 high3 token。")
            hf_summary = self.h3_project(h3)
            return hf_summary, {
                "hf_summary": hf_summary,
            }

        raise RuntimeError(f"未知高频聚合模式: {self.mode}。")


class HighToLowAlignedFusion(nn.Module):
    """
    HLAF:
    High-to-Low Aligned Fusion.

    当前修正版：
    - HF summary: [B, Dhf, 2, 2]
    - low token:  [B, Dlow, 1, 1]
    - H3 -> low 与 H2 -> H3 采用一致的 child-to-parent 思路；
    English: - H3 -> low H2 -> H3 child-to-parent ;
    - HF summary 通过 PixelUnshuffle(2) 折叠为 1×1 父级摘要；
    English: - HF summary PixelUnshuffle(2) 1×1 ;
    - 再与 low token 融合，并由 CrossPatchEncoder 做最终位置交互。

    Notes:
    - 这里不再使用 PACEBlock 将高频摘要扩展至 8×8；
    English: - PACEBlock 8×8;
    - low 层经 PCASE 后为 8×8 source，并作为一个整体块编码为 1×1 low token。
    English: - low PCASE 8×8 source, 1×1 low token.
    """

    def __init__(
        self,
        dhf: int,
        dlow: int,
        dimg: int,
        cpe_attn_dim: Optional[int] = 64,
        cpe_heads: int = 2,
        cpe_ffn_ratio: float = 1.5,
        dropout: float = 0.0,
    ):
        """
        初始化高频到低频的 1x1 对齐融合模块。
        English: 1x1 .

        输入:
        English: Input:
            dhf: 高频摘要通道。
            English: dhf: .
            dlow: low token 通道。
            English: dlow: low token .
            dimg: 最终图像 token 通道。
            English: dimg: image token .
            cpe_*: 最终 CrossPatchEncoder 配置。
            English: cpe_*: CrossPatchEncoder configuration.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.hf_to_low = nn.PixelUnshuffle(downscale_factor=2)
        self.fuse = FusionBlock(4 * dhf + dlow, dimg)
        self.final_cpe = CrossPatchEncoder(dimg, cpe_attn_dim, cpe_heads, cpe_ffn_ratio, dropout)

    def forward(self, hf_summary: torch.Tensor, low_token: torch.Tensor) -> torch.Tensor:
        """
        将 2x2 高频摘要折叠到 1x1，并与 low token 融合。
        English: 2x2 1x1, low token .

        输入:
            hf_summary: [B, Dhf, 2, 2]。
            low_token: [B, Dlow, 1, 1]。
        输出:
        English: Output:
            [B, Dimg, 1, 1] 图像融合 token。
            English: [B, Dimg, 1, 1] image token.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if hf_summary.shape[-2:] != (2, 2):
            raise ValueError(f"HF summary 应为 2×2，当前为 {tuple(hf_summary.shape[-2:])}。")
        if low_token.shape[-2:] != (1, 1):
            raise ValueError(f"low token 应为 1×1，当前为 {tuple(low_token.shape[-2:])}。")

        hf_parent = self.hf_to_low(hf_summary)
        x = torch.cat([hf_parent, low_token], dim=1)
        x = self.fuse(x)
        x = self.final_cpe(x)
        return x

# =============================================================================
# 6. Image branch
# =============================================================================

class PCHFNetImageBranch(nn.Module):
    """
    PC-HFNet 图像分支。
    Image branch of PC-HFNet.
    """

    def __init__(
        self,
        pca_priors: Dict,
        freeze_pca: bool = True,
        expected_image_hw: Optional[Tuple[int, int]] = (1024, 1024),
        input_feature_vector_ratio: float = 0.045,
        allocation_source: str = "eigvals",
        patch_size: int = 8,
        token_compression_ratio: float = 8.0,
        token_dim_min: int = 96,
        token_dim_round_multiple: int = 16,
        d1: Optional[int] = None,
        d2: Optional[int] = None,
        d3: Optional[int] = None,
        dh2: Optional[int] = None,
        dhf: Optional[int] = None,
        dlow: Optional[int] = None,
        dimg: Optional[int] = None,
        image_embed_dim: int = 24,
        ld_attn_dim: Optional[int] = 64,
        ld_heads: int = 2,
        cpe_attn_dim: Optional[int] = 64,
        cpe_heads: int = 2,
        ffn_ratio: float = 1.5,
        dropout: float = 0.0,
        structure: Optional[Union[str, Sequence[str]]] = "high1+high2+high3+low",
    ):
        """
        初始化 MFPC-HFNet 图像分支完整拓扑。
        English: MFPC-HFNet image.

        输入:
        English: Input:
            pca_priors: 离线 pca_priors_full.pt 读取后的先验字典。
            English: pca_priors: pca_priors_full.pt readdictionary.
            freeze_pca: 是否冻结固定 PCA 投影。
            English: freeze_pca: PCA .
            expected_image_hw: 期望图像尺寸；None 时不检查。
            English: expected_image_hw: image; None check.
            input_feature_vector_ratio / allocation_source: PCASE 容量和通道分配兼容参数。
            English: allocation_source: PCASE 容量和通道分配兼容参数.
            patch_size / token_compression_ratio / token_dim_*: LD token 宽度自适应控制。
            English: token_compression_ratio / token_dim_*: LD token 宽度自适应控制.
            d1/d2/d3/dh2/dhf/dlow/dimg: 可选显式覆盖的 token 宽度。
            English: d1/d2/d3/dh2/dhf/dlow/dimg: optionalexplicit token .
            image_embed_dim: 图像分支最终嵌入维度。
            English: image_embed_dim: image.
            ld_* / cpe_* / ffn_ratio / dropout: LD Encoder 和跨 patch 编码器配置。
            English: cpe_* / ffn_ratio / dropout: LD Encoder 和跨 patch 编码器配置.
            structure: 菜单传入的结构消融标签，控制启用 high1/high2/high3/low 中哪些频层。
            English: structure: menulabel, high1/high2/high3/low .

        输出:
        English: Output:
            构造完成后，forward(image) 输出 [B, image_embed_dim]。
            English: , forward(image) Output [B, image_embed_dim].

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.expected_image_hw = expected_image_hw
        self.input_feature_vector_ratio = float(input_feature_vector_ratio)
        self.patch_size = int(patch_size)
        self.token_compression_ratio = float(token_compression_ratio)
        self.token_dim_min = int(token_dim_min)
        self.token_dim_round_multiple = int(token_dim_round_multiple)
        self.structure_bands = normalize_mfpchf_structure(structure)
        self.active_high_bands = tuple(band for band in self.structure_bands if band != "low")
        self.has_high_frequency = bool(self.active_high_bands)

        if self.token_compression_ratio <= 0:
            raise ValueError(f"token_compression_ratio 必须为正数，当前为 {token_compression_ratio}。")
        if self.token_dim_min <= 0:
            raise ValueError(f"token_dim_min 必须为正整数，当前为 {token_dim_min}。")
        if self.token_dim_round_multiple <= 0:
            raise ValueError(f"token_dim_round_multiple 必须为正整数，当前为 {token_dim_round_multiple}。")

        pca_priors = _normalize_prior_root(pca_priors)
        self._validate_priors(pca_priors)

        self.high1_prior = (
            BandPriorProjection(pca_priors["high1"], "high1", freeze_pca=freeze_pca)
            if "high1" in self.structure_bands else None
        )
        self.high2_prior = (
            BandPriorProjection(pca_priors["high2"], "high2", freeze_pca=freeze_pca)
            if "high2" in self.structure_bands else None
        )
        self.high3_prior = (
            BandPriorProjection(pca_priors["high3"], "high3", freeze_pca=freeze_pca)
            if "high3" in self.structure_bands else None
        )
        self.low_prior = BandPriorProjection(pca_priors["low"], "low", freeze_pca=freeze_pca)

        self.high1_pcase = None
        if self.high1_prior is not None:
            self.high1_pcase = PrincipalComponentAdaptiveScaleEmbedding(
                self.high1_prior.num_components,
                self.high1_prior.eigvals,
                input_size=1024,
                target_size=64,
                input_feature_vector_ratio=self.input_feature_vector_ratio,
                min_dim=1,
                allocation_source=allocation_source,
                explained_variance_ratio=self.high1_prior.explained_variance_ratio,
                prior_feature_vector_ratio=self.high1_prior.effective_feature_vector_ratio,
            )
        self.high2_pcase = None
        if self.high2_prior is not None:
            self.high2_pcase = PrincipalComponentAdaptiveScaleEmbedding(
                self.high2_prior.num_components,
                self.high2_prior.eigvals,
                input_size=512,
                target_size=32,
                input_feature_vector_ratio=self.input_feature_vector_ratio,
                min_dim=1,
                allocation_source=allocation_source,
                explained_variance_ratio=self.high2_prior.explained_variance_ratio,
                prior_feature_vector_ratio=self.high2_prior.effective_feature_vector_ratio,
            )
        self.high3_pcase = None
        if self.high3_prior is not None:
            self.high3_pcase = PrincipalComponentAdaptiveScaleEmbedding(
                self.high3_prior.num_components,
                self.high3_prior.eigvals,
                input_size=256,
                target_size=16,
                input_feature_vector_ratio=self.input_feature_vector_ratio,
                min_dim=1,
                allocation_source=allocation_source,
                explained_variance_ratio=self.high3_prior.explained_variance_ratio,
                prior_feature_vector_ratio=self.high3_prior.effective_feature_vector_ratio,
            )
        self.low_pcase = PrincipalComponentAdaptiveScaleEmbedding(
            self.low_prior.num_components,
            self.low_prior.eigvals,
            input_size=128,
            target_size=8,
            input_feature_vector_ratio=self.input_feature_vector_ratio,
            min_dim=1,
            allocation_source=allocation_source,
            explained_variance_ratio=self.low_prior.explained_variance_ratio,
            # low 层不执行结构向量筛选，因此有效特征向量保留率固定为 1.0。
            # EN: low not executestructural-vector selection, therefore amount keep fixed as 1.0.
            # The low band is not structurally screened, so its effective vector ratio is fixed at 1.0.
            prior_feature_vector_ratio=1.0,
            # low 频层按常规 CNN 口径处理：空间边长每压缩 2 倍，通道约翻倍。
            # EN: low frequency band by CNN interface: each 2, pass.
            # For the low band, use conv-like channel growth instead of area-level expansion.
            capacity_mode="conv_like_channel_doubling",
        )
        self.low_pcase.feature_vector_ratio_source = "fixed_low_no_vector_screening"

        # ================= 自适应 token 宽度推导 =================
        # EN: ================= should token degree =================.
        # 逻辑：
        # EN: Logic:
        # 1. 各频层先按 PCASE 输出通道和 8×8 patch 容量计算自身 token 宽度；
        # EN: each frequency band first by PCASE pass and 8x8 patch amount token degree;
        # 2. D1/D2/D3/DLOW 分别对应 high1/high2/high3/low 的局部 token；
        # EN: D1/D2/D3/DLOW for should high1/high2/high3/low token;
        # 3. DH2 与 H2 尺度摘要对齐，默认继承 D2；DHF 与 H3 尺度摘要对齐，默认继承 D3；
        # EN: DH2 and H2 degree need alignment, default D2; DHF and H3 degree need alignment, default D3;
        # 4. DIMG 与最终 low 对齐摘要对齐，默认继承 DLOW；
        # EN: DIMG and most low alignment need alignment, default DLOW;
        # 5. d1/d2/d3/dh2/dhf/dlow/dimg 仍保留为可选显式覆盖接口，但不再是主推荐入口。
        # EN: d1/d2/d3/dh2/dhf/dlow/dimg still keep as optionalexplicit override interface, no longer is interface.
        derived_d1 = None if self.high1_pcase is None else derive_token_dim_from_patch_capacity(
            self.high1_pcase.total_out_channels, self.patch_size,
            self.token_compression_ratio, self.token_dim_min, self.token_dim_round_multiple
        )
        derived_d2 = None if self.high2_pcase is None else derive_token_dim_from_patch_capacity(
            self.high2_pcase.total_out_channels, self.patch_size,
            self.token_compression_ratio, self.token_dim_min, self.token_dim_round_multiple
        )
        derived_d3 = None if self.high3_pcase is None else derive_token_dim_from_patch_capacity(
            self.high3_pcase.total_out_channels, self.patch_size,
            self.token_compression_ratio, self.token_dim_min, self.token_dim_round_multiple
        )
        derived_dlow = derive_token_dim_from_patch_capacity(
            self.low_pcase.total_out_channels, self.patch_size,
            self.token_compression_ratio, self.token_dim_min, self.token_dim_round_multiple
        )

        self.d1 = int(derived_d1 if d1 is None else d1) if derived_d1 is not None else 0
        self.d2 = int(derived_d2 if d2 is None else d2) if derived_d2 is not None else 0
        self.d3 = int(derived_d3 if d3 is None else d3) if derived_d3 is not None else 0
        self.dlow = int(derived_dlow if dlow is None else dlow)
        self.dh2 = int(self.d2 if dh2 is None else dh2)
        self.dhf = int(self.d3 if dhf is None else dhf)
        self.dimg = int(self.dlow if dimg is None else dimg)

        self.token_dim_derivation = {
            "h1": (
                {"enabled": False, "status": "not_instantiated_in_this_ablation_mode"}
                if self.high1_pcase is None
                else summarize_token_dim_derivation(self.high1_pcase.total_out_channels, self.patch_size, self.d1, self.token_compression_ratio)
            ),
            "h2": (
                {"enabled": False, "status": "not_instantiated_in_this_ablation_mode"}
                if self.high2_pcase is None
                else summarize_token_dim_derivation(self.high2_pcase.total_out_channels, self.patch_size, self.d2, self.token_compression_ratio)
            ),
            "h3": (
                {"enabled": False, "status": "not_instantiated_in_this_ablation_mode"}
                if self.high3_pcase is None
                else summarize_token_dim_derivation(self.high3_pcase.total_out_channels, self.patch_size, self.d3, self.token_compression_ratio)
            ),
            "low": summarize_token_dim_derivation(self.low_pcase.total_out_channels, self.patch_size, self.dlow, self.token_compression_ratio),
        }
        self.token_dim_override_flags = {
            "d1": d1 is not None,
            "d2": d2 is not None,
            "d3": d3 is not None,
            "dh2": dh2 is not None,
            "dhf": dhf is not None,
            "dlow": dlow is not None,
            "dimg": dimg is not None,
        }

        self.h1_encoder = None
        if self.high1_pcase is not None:
            self.h1_encoder = PatchGridLDEncoder(
                self.high1_pcase.total_out_channels,
                self.d1,
                patch_size=patch_size,
                attn_dim=ld_attn_dim,
                num_heads=ld_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
        self.h2_encoder = None
        if self.high2_pcase is not None:
            self.h2_encoder = PatchGridLDEncoder(
                self.high2_pcase.total_out_channels,
                self.d2,
                patch_size=patch_size,
                attn_dim=ld_attn_dim,
                num_heads=ld_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
        self.h3_encoder = None
        if self.high3_pcase is not None:
            self.h3_encoder = PatchGridLDEncoder(
                self.high3_pcase.total_out_channels,
                self.d3,
                patch_size=patch_size,
                attn_dim=ld_attn_dim,
                num_heads=ld_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
        self.low_encoder = PatchGridLDEncoder(
            self.low_pcase.total_out_channels,
            self.dlow,
            patch_size=patch_size,
            attn_dim=ld_attn_dim,
            num_heads=ld_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
        )

        self.hfsa = None
        self.hlaf = None
        if self.has_high_frequency:
            self.hfsa = StructureAwareHighFrequencyAggregation(
                active_high_bands=self.active_high_bands,
                d1=self.d1,
                d2=self.d2,
                d3=self.d3,
                dh2=self.dh2,
                dhf=self.dhf,
                cpe_attn_dim=cpe_attn_dim,
                cpe_heads=cpe_heads,
                cpe_ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
            self.hlaf = HighToLowAlignedFusion(
                dhf=self.dhf,
                dlow=self.dlow,
                dimg=self.dimg,
                cpe_attn_dim=cpe_attn_dim,
                cpe_heads=cpe_heads,
                cpe_ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
            self.low_only_project = nn.Identity()
        else:
            self.low_only_project = nn.Identity() if self.dimg == self.dlow else nn.Conv2d(self.dlow, self.dimg, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        hidden = max(image_embed_dim * 2, image_embed_dim)
        self.image_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dimg, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Dropout(0.0),
            nn.Linear(hidden, image_embed_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _validate_priors(pca_priors: Dict):
        """
        检查先验字典是否包含四个必需频层。
        English: checkdictionary.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        required_bands = ["high1", "high2", "high3", "low"]
        missing = [b for b in required_bands if b not in pca_priors]
        if missing:
            raise KeyError(f"pca_priors 缺少频层: {missing}")

    def _check_image_size(self, image: torch.Tensor):
        """
        校验输入图像空间尺寸是否符合当前菜单/模型预期。
        English: validationInputimagecurrentmenu/model.

        输入:
            image: [B, 8, H, W]。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if self.expected_image_hw is None:
            return
        expected_h, expected_w = int(self.expected_image_hw[0]), int(self.expected_image_hw[1])
        cur_h, cur_w = int(image.shape[-2]), int(image.shape[-1])
        if (cur_h, cur_w) != (expected_h, expected_w):
            raise ValueError(f"PC-HFNet 期望图像尺寸 {(expected_h, expected_w)}，实际 {(cur_h, cur_w)}。")

    def get_structure_summary(self) -> Dict:
        """
        返回模型图像分支的关键结构参数，便于打印、manifest 记录和后续论文核对。
        Return image-branch structural parameters for logging and manifest export.
        """
        expected_hw = None
        if self.expected_image_hw is not None:
            expected_hw = [int(self.expected_image_hw[0]), int(self.expected_image_hw[1])]

        band_summary = {
            "high1": summarize_pcase_band("high1", self.high1_prior, self.high1_pcase, enabled=True),
            "high2": summarize_pcase_band("high2", self.high2_prior, self.high2_pcase, enabled=True),
            "high3": summarize_pcase_band("high3", self.high3_prior, self.high3_pcase, enabled=True),
            "low": summarize_pcase_band("low", self.low_prior, self.low_pcase, enabled=True),
        }

        token_summary = {
            "token_dim_source": "adaptive_fixed_compression_ratio",
            "token_compression_ratio": float(self.token_compression_ratio),
            "token_dim_min": int(self.token_dim_min),
            "token_dim_round_multiple": int(self.token_dim_round_multiple),
            "token_dim_override_flags": dict(self.token_dim_override_flags),
            "per_band_derivation": dict(self.token_dim_derivation),
            "resolved_token_dims": {
                "d1": int(self.d1),
                "d2": int(self.d2),
                "d3": int(self.d3),
                "dh2": int(self.dh2),
                "dhf": int(self.dhf),
                "dlow": int(self.dlow),
                "dimg": int(self.dimg),
            },
            "fusion_dim_policy": {
                "dh2": "d2 unless explicitly overridden",
                "dhf": "d3 unless explicitly overridden",
                "dimg": "dlow unless explicitly overridden",
            },
        }

        summary = {
            "expected_image_hw": expected_hw,
            "structure_bands": list(self.structure_bands),
            "active_high_bands": list(self.active_high_bands),
            "patch_size": int(self.patch_size),
            "fallback_input_feature_vector_ratio": float(self.input_feature_vector_ratio),
            "pcase_capacity_summary": band_summary,
            "token_dimension_summary": token_summary,
        }

        # 兼容旧日志字段：保留原来的扁平键，避免已有分析脚本读取失败。
        # EN: compatible old field: keep, avoid already read.
        for band_name, item in band_summary.items():
            summary[f"{band_name}_effective_feature_vector_ratio"] = item.get("effective_feature_vector_ratio")
            summary[f"{band_name}_feature_vector_ratio_source"] = item.get("feature_vector_ratio_source")
            summary[f"{band_name}_pcase_channels"] = item.get("total_out_channels")
            summary[f"{band_name}_branch_dims"] = item.get("branch_dims")
            summary[f"{band_name}_capacity_mode"] = item.get("capacity_mode")
            summary[f"{band_name}_channel_scale_factor_source"] = item.get("channel_scale_factor_source")
            summary[f"{band_name}_raw_channel_scale_factor_used"] = item.get("raw_channel_scale_factor_used")
            summary[f"{band_name}_raw_channel_scale_factor_s_times_nl"] = item.get("raw_channel_scale_factor_s_times_nl")
            summary[f"{band_name}_raw_channel_scale_factor_s2_times_nl"] = item.get("raw_channel_scale_factor_s2_times_nl")
            summary[f"{band_name}_raw_total_channels_before_ceil"] = item.get("raw_total_channels_before_ceil")
            summary[f"{band_name}_realized_channel_scale_factor"] = item.get("realized_channel_scale_factor")

        # 兼容旧日志字段：同时写出当前实际 token 宽度。
        # EN: compatible old field: same when write current token degree.
        summary["h1_token_dim"] = int(self.d1)
        summary["h2_token_dim"] = int(self.d2)
        summary["h3_token_dim"] = int(self.d3)
        summary["hf2_token_dim"] = int(self.dh2)
        summary["hf_token_dim"] = int(self.dhf)
        summary["low_token_dim"] = int(self.dlow)
        summary["image_token_dim"] = int(self.dimg)

        return summary

    def forward(self, image: torch.Tensor, return_aux: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        执行图像分支前向传播。
        English: image.

        输入:
            image: [B, 8, H, W]。
            return_aux: True 时额外返回 PCA、PCASE、token grid 和融合中间量。
            English: return_aux: True return PCA, PCASE, token grid .
        输出:
            return_aux=False: [B, image_embed_dim]。
            return_aux=True: (image embedding, aux dict)。

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        self._check_image_size(image)

        pyramid = build_structure_laplacian_pyramid(image, self.structure_bands)

        high1_pc = self.high1_prior(pyramid["high1"]) if self.high1_prior is not None else None
        high2_pc = self.high2_prior(pyramid["high2"]) if self.high2_prior is not None else None
        high3_pc = self.high3_prior(pyramid["high3"]) if self.high3_prior is not None else None
        low_pc = self.low_prior(pyramid["low"])

        f1 = self.high1_pcase(high1_pc) if self.high1_pcase is not None else None
        f2 = self.high2_pcase(high2_pc) if self.high2_pcase is not None else None
        f3 = self.high3_pcase(high3_pc) if self.high3_pcase is not None else None
        low_source = self.low_pcase(low_pc)

        h1 = self.h1_encoder(f1) if self.h1_encoder is not None else None
        h2 = self.h2_encoder(f2) if self.h2_encoder is not None else None
        h3 = self.h3_encoder(f3) if self.h3_encoder is not None else None
        low_token = self.low_encoder(low_source)

        hfsa_aux: Dict[str, torch.Tensor] = {}
        hf_summary = None
        if self.has_high_frequency:
            hf_summary, hfsa_aux = self.hfsa(h1, h2, h3)
            enhanced_tokens = self.hlaf(hf_summary, low_token)
        else:
            enhanced_tokens = self.low_only_project(low_token)

        img_feat = self.image_head(self.global_pool(enhanced_tokens))

        if not return_aux:
            return img_feat

        aux = {
            "low_pc": low_pc,
            "low_source": low_source,
            "low_token": low_token,
            "enhanced_tokens": enhanced_tokens,
        }
        if high1_pc is not None:
            aux["high1_pc"] = high1_pc
            aux["f1_source"] = f1
            aux["h1_grid"] = h1
        if high2_pc is not None:
            aux["high2_pc"] = high2_pc
            aux["f2_source"] = f2
            aux["h2_grid"] = h2
        if high3_pc is not None:
            aux["high3_pc"] = high3_pc
            aux["f3_source"] = f3
            aux["h3_grid"] = h3
        if hf_summary is not None:
            aux["hf_summary"] = hf_summary
        aux.update(hfsa_aux)
        return img_feat, aux


# =============================================================================
# 7. Hyper / NIR branches and full multimodal model
# =============================================================================

class HyperBranch(nn.Module):
    """
    高光谱分支，输入 [B, 681]。
    English: , Input [B, 681].
    """

    def __init__(self, in_dim: int = 681, embed_dim: int = 32):
        """
        初始化 HyperVISNIR 数值分支。
        English: HyperVISNIR .

        输入:
        English: Input:
            in_dim: HyperVISNIR 输入维度，默认 681。
            English: in_dim: HyperVISNIR Input, default 681.
            embed_dim: 输出嵌入维度。
            English: embed_dim: Output.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 HyperVISNIR 特征映射为低维嵌入。
        English: HyperVISNIR .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return self.net(x)


class NIRBranch(nn.Module):
    """
    NIR 低维特征分支，输入 [B, 5]。
    English: NIR , Input [B, 5].
    """

    def __init__(self, in_dim: int = 5, embed_dim: int = 16):
        """
        初始化 NIR 数值分支。
        English: NIR .

        输入:
        English: Input:
            in_dim: NIR 输入维度，默认 5。
            English: in_dim: NIR Input, default 5.
            embed_dim: 输出嵌入维度。
            English: embed_dim: Output.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(inplace=True),
            nn.LayerNorm(16),
            nn.Linear(16, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 NIR 特征映射为低维嵌入。
        English: NIR .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        return self.net(x)


VALID_ACTIVE_INPUTS = ("image", "hyper", "nir")


def normalize_active_inputs(active_inputs: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """
    规范化输入端消融启用分支。
    English: normalizeInput.

    物理/工程意义：
    English: /:
    - 菜单中的 active_inputs 是本次训练模型的输入组成；
    English: - menu active_inputs trainingmodelInput;
    - 该函数只负责清洗和校验，不自行增删菜单没有声明的输入源；
    English: - validation, menuInput;
    - 最近修改时间：2026-05-29；作者：ljy。
    English: - Last modified: 2026-05-29; Author: ljy.
    """

    if active_inputs is None:
        return VALID_ACTIVE_INPUTS

    normalized: List[str] = []
    for item in active_inputs:
        name = str(item).strip().lower()
        if not name:
            continue
        if name not in VALID_ACTIVE_INPUTS:
            raise ValueError(f"active_inputs 仅支持 {VALID_ACTIVE_INPUTS}，当前包含: {item!r}")
        if name not in normalized:
            normalized.append(name)

    if not normalized:
        raise ValueError("active_inputs 至少需要包含一个输入源。")
    return tuple(normalized)


def split_transformer_fusion_optimizer_groups(model: nn.Module) -> list[dict]:
    """
    按 MFPC-HFNet 融合层微调语义拆分 optimizer 参数组。
    English: MFPC-HFNet optimizer parameter.

    逻辑 / Logic:
    English: Logic:.
    1. `transformer_fusion` 组对应历史菜单中的融合/合并层低 LR 对象：
    English: 1. `transformer_fusion` menu/ LR :
       `image_branch.hfsa.cpe_hf2`、`image_branch.hfsa.cpe_hf` 和 `image_branch.hlaf.final_cpe`；
       English: `image_branch.hfsa.cpe_hf2`, `image_branch.hfsa.cpe_hf` `image_branch.hlaf.final_cpe`;
    2. `base` 组包含其余所有参数，保持菜单基础学习率；
    English: 2. `base` parameter, menulearning rate;
    3. 本函数只提供模型内部参数语义，不创建 optimizer，也不决定具体学习率。
    English: 3. modelparameter, create optimizer, learning rate.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    fusion_keywords = (
        "image_branch.hfsa.cpe_hf2",
        "image_branch.hfsa.cpe_hf",
        "image_branch.hlaf.final_cpe",
    )
    base_params = []
    fusion_params = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in fusion_keywords):
            fusion_params.append(param)
        else:
            base_params.append(param)

    groups = []
    if base_params:
        groups.append({
            "name": "base",
            "lr_role": "default",
            "params": base_params,
        })
    if fusion_params:
        groups.append({
            "name": "transformer_fusion",
            "lr_role": "transformer_fusion",
            "params": fusion_params,
        })
    return groups


class SOCInversionModel(nn.Module):
    """
    PC-HFNet 多模态回归模型。
    English: PC-HFNet model.

    保持现有训练脚本接口：
        output = model(hyper, nir, image)

    Single target:
        output_dim=1 -> [B]
    Two targets:
        output_dim=2 -> [B, 2]
    """

    def __init__(
        self,
        feature_dim_hyper: int = 681,
        feature_dim_nir: int = 5,
        output_dim: int = 1,
        pca_priors_path: Optional[str] = None,
        pca_priors: Optional[Dict] = None,
        freeze_pca: bool = True,
        expected_image_hw: Optional[Tuple[int, int]] = (1024, 1024),
        input_feature_vector_ratio: float = 0.045,
        allocation_source: str = "eigvals",
        image_embed_dim: int = 24,
        hyper_embed_dim: int = 32,
        nir_embed_dim: int = 16,
        fusion_hidden_dim: int = 48,
        token_compression_ratio: float = 8.0,
        token_dim_min: int = 96,
        token_dim_round_multiple: int = 16,
        d1: Optional[int] = None,
        d2: Optional[int] = None,
        d3: Optional[int] = None,
        dh2: Optional[int] = None,
        dhf: Optional[int] = None,
        dlow: Optional[int] = None,
        dimg: Optional[int] = None,
        ld_attn_dim: Optional[int] = 64,
        ld_heads: int = 2,
        cpe_attn_dim: Optional[int] = 64,
        cpe_heads: int = 2,
        ffn_ratio: float = 1.5,
        dropout: float = 0.0,
        laplacian_levels: int = 3,
        base_channels: Optional[int] = None,
        structure: Optional[Union[str, Sequence[str]]] = "high1+high2+high3+low",
    ):
        """
        初始化 Full 多模态 SOC/TN 回归模型。
        English: Full SOC/TN model.

        输入:
        English: Input:
            feature_dim_hyper / feature_dim_nir: 数值分支输入维度。
            English: feature_dim_nir: 数值分支输入维度.
            output_dim: 1 表示单目标，2 表示 SOC+TN 双目标。
            English: output_dim: 1 , 2 SOC+TN .
            pca_priors_path / pca_priors: 离线 PCA 先验来源，二选一。
            English: pca_priors: 离线 PCA 先验来源，二选一.
            freeze_pca / expected_image_hw / PCASE-token 参数: 传递给图像分支。
            English: expected_image_hw / PCASE-token 参数: 传递给图像分支.
            image_embed_dim / hyper_embed_dim / nir_embed_dim: 三个输入源的嵌入宽度。
            English: hyper_embed_dim / nir_embed_dim: 三个输入源的嵌入宽度.
            fusion_hidden_dim: 晚期融合回归头隐藏维度。
            English: fusion_hidden_dim: .
            laplacian_levels: 当前固定为 3，仅用于接口兼容。
            English: laplacian_levels: current 3, compatible.
            structure: 菜单声明的 MFPC-HFNet 结构消融标签。
            English: structure: menu MFPC-HFNet label.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.output_dim = int(output_dim)
        if self.output_dim not in (1, 2):
            raise ValueError(f"output_dim 仅支持 1 或 2，当前为 {self.output_dim}。")
        if int(laplacian_levels) != 3:
            raise ValueError("PC-HFNet 当前固定使用 laplacian_levels=3。")

        priors = self._load_priors(pca_priors_path=pca_priors_path, pca_priors=pca_priors)

        self.image_branch = PCHFNetImageBranch(
            pca_priors=priors,
            freeze_pca=freeze_pca,
            expected_image_hw=expected_image_hw,
            input_feature_vector_ratio=input_feature_vector_ratio,
            allocation_source=allocation_source,
            d1=d1,
            d2=d2,
            d3=d3,
            dh2=dh2,
            dhf=dhf,
            dlow=dlow,
            dimg=dimg,
            image_embed_dim=image_embed_dim,
            token_compression_ratio=token_compression_ratio,
            token_dim_min=token_dim_min,
            token_dim_round_multiple=token_dim_round_multiple,
            ld_attn_dim=ld_attn_dim,
            ld_heads=ld_heads,
            cpe_attn_dim=cpe_attn_dim,
            cpe_heads=cpe_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            structure=structure,
        )

        self.hyper_branch = HyperBranch(in_dim=feature_dim_hyper, embed_dim=hyper_embed_dim)
        self.nir_branch = NIRBranch(in_dim=feature_dim_nir, embed_dim=nir_embed_dim)

        fusion_in_dim = int(image_embed_dim) + int(hyper_embed_dim) + int(nir_embed_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_hidden_dim),
            nn.Dropout(0.15),
            nn.Linear(fusion_hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.output_dim),
        )

    @staticmethod
    def _load_priors(pca_priors_path: Optional[str] = None, pca_priors: Optional[Dict] = None) -> Dict:
        """
        从路径或内存对象读取 PCA 先验。
        English: pathread PCA .

        输入:
        English: Input:
            pca_priors_path: pca_priors_full.pt 路径。
            English: pca_priors_path: pca_priors_full.pt path.
            pca_priors: 已经加载到内存的先验字典。
            English: pca_priors: loaddictionary.
        输出:
        English: Output:
            规范化后的四频层先验字典。
            English: normalizedictionary.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if pca_priors is not None:
            return _normalize_prior_root(pca_priors)
        if pca_priors_path is None:
            raise ValueError("必须提供 pca_priors_path 或 pca_priors。")
        return _normalize_prior_root(_load_torch_file(pca_priors_path))

    def get_structure_summary(self) -> Dict:
        """
        返回图像主干关键结构信息，便于训练日志记录。
        English: returnimage, training.
        """
        return self.image_branch.get_structure_summary()

    def get_optimizer_parameter_groups(self) -> list[dict]:
        """
        返回 MFPC-HFNet 全输入模型的 optimizer 参数分组。
        English: return MFPC-HFNet Inputmodel optimizer parameter groups.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """

        return split_transformer_fusion_optimizer_groups(self)

    def forward(self, hyper: torch.Tensor, nir: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        执行 Full 多模态回归前向传播。
        English: Full .

        输入:
        English: Input:
            hyper: [B, 681] HyperVISNIR 特征。
            English: hyper: [B, 681] HyperVISNIR .
            nir: [B, 5] NIR 特征。
            English: nir: [B, 5] NIR .
            image: [B, 8, H, W] 图像。
            English: image: [B, 8, H, W] image.
        输出:
        English: Output:
            output_dim=1 时返回 [B]；output_dim=2 时返回 [B, 2]。
            English: output_dim=1 return [B]; output_dim=2 return [B, 2].

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        img_feat = self.image_branch(image, return_aux=False)
        hyper_feat = self.hyper_branch(hyper)
        nir_feat = self.nir_branch(nir)

        fused = torch.cat([img_feat, hyper_feat, nir_feat], dim=1)
        out = self.fusion_head(fused)

        if self.output_dim == 1:
            return out.view(-1)
        return out


class InputAblationSOCInversionModel(nn.Module):
    """
    输入端消融回归模型。
    English: Inputmodel.

    逻辑说明 / Logic:
    English: Logic:.
    1. active_inputs 由菜单直接传入，模型层只按该清单实例化 image / hyper / nir 分支；
    English: hyper / nir 分支；.
    2. 未启用的分支不会出现在 state_dict 中，保证续训 checkpoint 与模型结构一一对应；
    English: 2. state_dict , ensure checkpoint model;
    3. 融合头输入维度由启用分支嵌入维度求和得到，不在训练引擎中硬编码；
    English: 3. Input, training engine;
    4. 最近修改时间：2026-05-29；作者：ljy。
    English: 4. Last modified: 2026-05-29; Author: ljy.
    """

    def __init__(
        self,
        active_inputs: Sequence[str],
        feature_dim_hyper: int = 681,
        feature_dim_nir: int = 5,
        output_dim: int = 1,
        pca_priors_path: Optional[str] = None,
        pca_priors: Optional[Dict] = None,
        freeze_pca: bool = True,
        expected_image_hw: Optional[Tuple[int, int]] = (1024, 1024),
        input_feature_vector_ratio: float = 0.045,
        allocation_source: str = "eigvals",
        image_embed_dim: int = 24,
        hyper_embed_dim: int = 32,
        nir_embed_dim: int = 16,
        fusion_hidden_dim: int = 48,
        token_compression_ratio: float = 8.0,
        token_dim_min: int = 96,
        token_dim_round_multiple: int = 16,
        d1: Optional[int] = None,
        d2: Optional[int] = None,
        d3: Optional[int] = None,
        dh2: Optional[int] = None,
        dhf: Optional[int] = None,
        dlow: Optional[int] = None,
        dimg: Optional[int] = None,
        ld_attn_dim: Optional[int] = 64,
        ld_heads: int = 2,
        cpe_attn_dim: Optional[int] = 64,
        cpe_heads: int = 2,
        ffn_ratio: float = 1.5,
        dropout: float = 0.0,
        laplacian_levels: int = 3,
        structure: Optional[Union[str, Sequence[str]]] = "high1+high2+high3+low",
    ):
        """
        初始化输入端消融模型。
        English: Inputmodel.

        输入:
        English: Input:
            active_inputs: 本轮启用的输入源组合，例如 image/hyper/nir。
            English: active_inputs: Input, image/hyper/nir.
            其余参数与 SOCInversionModel 保持同名含义，用于构造启用的分支和融合头。
            English: parameter SOCInversionModel , .
            structure: 启用 image 分支时使用的 MFPC-HFNet 结构标签。
            English: structure: image MFPC-HFNet label.

        关键约束:
        English: :
            未启用分支不会实例化，因此 checkpoint 中不会出现无效分支参数。
            English: , checkpoint parameter.
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        super().__init__()

        self.active_inputs = normalize_active_inputs(active_inputs)
        self.output_dim = int(output_dim)
        if self.output_dim not in (1, 2):
            raise ValueError(f"output_dim 仅支持 1 或 2，当前为 {self.output_dim}。")
        if int(laplacian_levels) != 3:
            raise ValueError("PC-HFNet 当前固定使用 laplacian_levels=3。")

        fusion_in_dim = 0
        self.image_branch = None
        self.hyper_branch = None
        self.nir_branch = None

        if "image" in self.active_inputs:
            priors = SOCInversionModel._load_priors(pca_priors_path=pca_priors_path, pca_priors=pca_priors)
            self.image_branch = PCHFNetImageBranch(
                pca_priors=priors,
                freeze_pca=freeze_pca,
                expected_image_hw=expected_image_hw,
                input_feature_vector_ratio=input_feature_vector_ratio,
                allocation_source=allocation_source,
                d1=d1,
                d2=d2,
                d3=d3,
                dh2=dh2,
                dhf=dhf,
                dlow=dlow,
                dimg=dimg,
                image_embed_dim=image_embed_dim,
                token_compression_ratio=token_compression_ratio,
                token_dim_min=token_dim_min,
                token_dim_round_multiple=token_dim_round_multiple,
                ld_attn_dim=ld_attn_dim,
                ld_heads=ld_heads,
                cpe_attn_dim=cpe_attn_dim,
                cpe_heads=cpe_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                structure=structure,
            )
            fusion_in_dim += int(image_embed_dim)

        if "hyper" in self.active_inputs:
            self.hyper_branch = HyperBranch(in_dim=feature_dim_hyper, embed_dim=hyper_embed_dim)
            fusion_in_dim += int(hyper_embed_dim)

        if "nir" in self.active_inputs:
            self.nir_branch = NIRBranch(in_dim=feature_dim_nir, embed_dim=nir_embed_dim)
            fusion_in_dim += int(nir_embed_dim)

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_hidden_dim),
            nn.Dropout(0.15),
            nn.Linear(fusion_hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.output_dim),
        )

    def get_structure_summary(self) -> Dict:
        """
        返回当前输入端消融模型结构摘要。
        English: returncurrentInputmodel.
        """

        summary = {
            "active_inputs": list(self.active_inputs),
            "image_branch_enabled": "image" in self.active_inputs,
            "hyper_branch_enabled": "hyper" in self.active_inputs,
            "nir_branch_enabled": "nir" in self.active_inputs,
        }
        if self.image_branch is None:
            summary["image_branch"] = {"status": "not_instantiated_in_this_ablation_mode"}
        else:
            summary["image_branch"] = self.image_branch.get_structure_summary()
        return summary

    def get_optimizer_parameter_groups(self) -> list[dict]:
        """
        返回输入端消融模型的 optimizer 参数分组。
        English: returnInputmodel optimizer parameter groups.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        说明：若当前输入组合启用了 image 分支，则融合层低 LR 策略仍可复用；无 image 分支时只返回 base 组。
        English: : currentInput image , LR ; image return base .
        """

        return split_transformer_fusion_optimizer_groups(self)

    def forward(self, hyper: torch.Tensor, nir: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        按 active_inputs 收集启用分支特征并完成回归。
        English: active_inputs .

        输入:
        English: Input:
            hyper/nir/image: 为保持训练引擎统一接口仍全部保留；未启用分支不会读取对应输入。
            English: hyper/nir/image: training engine; readInput.
        输出:
        English: Output:
            output_dim=1 时返回 [B]；output_dim=2 时返回 [B, 2]。
            English: output_dim=1 return [B]; output_dim=2 return [B, 2].

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        features: List[torch.Tensor] = []

        if "image" in self.active_inputs:
            if image is None:
                raise ValueError("当前 active_inputs 启用了 image，但 forward 未收到 image。")
            features.append(self.image_branch(image, return_aux=False))

        if "hyper" in self.active_inputs:
            if hyper is None:
                raise ValueError("当前 active_inputs 启用了 hyper，但 forward 未收到 hyper。")
            features.append(self.hyper_branch(hyper))

        if "nir" in self.active_inputs:
            if nir is None:
                raise ValueError("当前 active_inputs 启用了 nir，但 forward 未收到 nir。")
            features.append(self.nir_branch(nir))

        fused = torch.cat(features, dim=1)
        out = self.fusion_head(fused)

        if self.output_dim == 1:
            return out.view(-1)
        return out


def build_model(
    pca_priors_path: Optional[str],
    output_dim: int = 1,
    feature_dim_hyper: int = 681,
    feature_dim_nir: int = 5,
    expected_image_hw: Tuple[int, int] = (1024, 1024),
    input_feature_vector_ratio: float = 0.045,
    allocation_source: str = "eigvals",
    freeze_pca: bool = True,
    image_embed_dim: int = 24,
    hyper_embed_dim: int = 32,
    nir_embed_dim: int = 16,
    fusion_hidden_dim: int = 48,
    token_compression_ratio: float = 8.0,
    token_dim_min: int = 96,
    token_dim_round_multiple: int = 16,
    ld_attn_dim: Optional[int] = 64,
    ld_heads: int = 2,
    cpe_attn_dim: Optional[int] = 64,
    cpe_heads: int = 2,
    ffn_ratio: float = 1.5,
    dropout: float = 0.0,
    structure: Optional[Union[str, Sequence[str]]] = "high1+high2+high3+low",
    active_inputs: Optional[Sequence[str]] = None,
) -> nn.Module:
    """
    便捷构造函数。
    English: This docstring documents the corresponding function behavior and engineering constraints.

    输入:
    English: Input:
        structure: 菜单传入的结构消融标签，用于控制图像分支启用的频层组合。
        English: structure: menulabel, image.
        active_inputs: 输入端消融标签，用于控制 image/hyper/nir 分支是否实例化。
        English: active_inputs: Inputlabel, image/hyper/nir .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    normalized_inputs = normalize_active_inputs(active_inputs)
    if normalized_inputs != VALID_ACTIVE_INPUTS:
        return InputAblationSOCInversionModel(
            active_inputs=normalized_inputs,
            feature_dim_hyper=feature_dim_hyper,
            feature_dim_nir=feature_dim_nir,
            output_dim=output_dim,
            pca_priors_path=pca_priors_path,
            freeze_pca=freeze_pca,
            expected_image_hw=expected_image_hw,
            input_feature_vector_ratio=input_feature_vector_ratio,
            allocation_source=allocation_source,
            image_embed_dim=image_embed_dim,
            hyper_embed_dim=hyper_embed_dim,
            nir_embed_dim=nir_embed_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            token_compression_ratio=token_compression_ratio,
            token_dim_min=token_dim_min,
            token_dim_round_multiple=token_dim_round_multiple,
            ld_attn_dim=ld_attn_dim,
            ld_heads=ld_heads,
            cpe_attn_dim=cpe_attn_dim,
            cpe_heads=cpe_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            structure=structure,
        )

    return SOCInversionModel(
        feature_dim_hyper=feature_dim_hyper,
        feature_dim_nir=feature_dim_nir,
        output_dim=output_dim,
        pca_priors_path=pca_priors_path,
        freeze_pca=freeze_pca,
        expected_image_hw=expected_image_hw,
        input_feature_vector_ratio=input_feature_vector_ratio,
        allocation_source=allocation_source,
        image_embed_dim=image_embed_dim,
        hyper_embed_dim=hyper_embed_dim,
        nir_embed_dim=nir_embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        token_compression_ratio=token_compression_ratio,
        token_dim_min=token_dim_min,
        token_dim_round_multiple=token_dim_round_multiple,
        ld_attn_dim=ld_attn_dim,
        ld_heads=ld_heads,
        cpe_attn_dim=cpe_attn_dim,
        cpe_heads=cpe_heads,
        ffn_ratio=ffn_ratio,
        dropout=dropout,
        structure=structure,
    )


