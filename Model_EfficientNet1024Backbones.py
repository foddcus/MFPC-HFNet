# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
EfficientNet 1024x1024 backbone builders for comparison training.

Generated/Updated: 2026-05-22

Design outline / 设计大纲：
1. EfficientNet V1 1024 follows the compound scaling equation from
   EfficientNet: depth=alpha^phi, width=beta^phi, resolution=gamma^phi.
   这里按论文的复合缩放公式由 B0 的 224 输入外推到 1024 输入。
2. EfficientNetV2 1024 uses the official V2-XL scale level as the 1024
   backbone template. Compared with V2-L, V2-XL increases later-stage depth
   and width non-uniformly, matching the V2 paper's scaling strategy while
   avoiding an impractically large extra extrapolation beyond XL.
   这里以官方 V2-XL 作为 1024 输入的深宽模板，相比 V2-L 已同步提高深度和宽度，避免只改输入分辨率。
3. Both builders return torchvision.models.efficientnet.EfficientNet objects.
   The training script will adapt the first convolution from RGB to the local
   8-channel image tensor and remove the classification head.
4. The two 1024 models are trained from scratch with micro-batch=1 in the
   compare runner. The backbone keeps the default EfficientNet BatchNorm
   structure, while the training loop freezes BatchNorm running statistics for
   tiny micro-batches.
"""

from __future__ import annotations

import math
from functools import partial
from typing import List, Tuple, Union

from torchvision.models.efficientnet import EfficientNet, FusedMBConvConfig, MBConvConfig


# ================= EfficientNet V1 复合缩放参数 / V1 compound scaling constants =================
# EN: ================= EfficientNet V1 parameters / V1 compound scaling constants =================.
EFFICIENTNET_V1_ALPHA = 1.2
EFFICIENTNET_V1_BETA = 1.1
EFFICIENTNET_V1_GAMMA = 1.15
EFFICIENTNET_V1_BASE_RESOLUTION = 224
EFFICIENTNET_1024_RESOLUTION = 1024

def get_efficientnet_v1_1024_scaling() -> Tuple[float, float, float]:
    """
    Calculate EfficientNet V1 compound scaling for 1024 input.

    物理/算法含义：
    English: /:
    - phi 由目标分辨率和基准分辨率决定：1024 = 224 * gamma^phi；
    English: - phi : 1024 = 224 * gamma^phi;
    - width_mult = beta^phi，同步增加每个 stage 的通道宽度；
    English: - width_mult = beta^phi, stage ;
    - depth_mult = alpha^phi，同步增加每个 stage 的重复层数；
    English: - depth_mult = alpha^phi, stage ;
    - 返回值用于 torchvision 的 MBConvConfig，实际通道数仍会按 8 对齐。

    Returns:
        (phi, width_mult, depth_mult)
    """
    phi = math.log(EFFICIENTNET_1024_RESOLUTION / EFFICIENTNET_V1_BASE_RESOLUTION) / math.log(
        EFFICIENTNET_V1_GAMMA
    )
    width_mult = EFFICIENTNET_V1_BETA ** phi
    depth_mult = EFFICIENTNET_V1_ALPHA ** phi
    return float(phi), float(width_mult), float(depth_mult)


def build_efficientnet_v1_1024(num_classes: int = 1000) -> EfficientNet:
    """
    Build a custom EfficientNet V1 backbone for 1024x1024 input.

    说明：
    English: :
    - 该结构不是 torchvision 官方预训练型号，因此不加载预训练权重；
    English: - torchvision training, loadtraining;
    - 层操作仍保持 EfficientNet-B0 的 MBConv stage 设计，只按论文复合缩放放大深度和宽度；
    English: - EfficientNet-B0 MBConv stage , ;
    - dropout 沿用 B7/B8/L2 一类大模型常用的 0.5，训练脚本外部会替换分类头；
    English: - dropout B7/B8/L2 model 0.5, training;
    - 1024 模型保留 EfficientNet 默认 BatchNorm；训练脚本在 micro-batch 很小时冻结
    English: - 1024 model EfficientNet default BatchNorm; training micro-batch.
      BatchNorm running statistics，避免小批次反复污染统计量；
      English: BatchNorm running statistics, avoid;
    - 最近修改时间：2026-05-22。
    English: - Last modified: 2026-05-22.
    """
    _, width_mult, depth_mult = get_efficientnet_v1_1024_scaling()
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    return EfficientNet(
        inverted_residual_setting,
        dropout=0.5,
        stochastic_depth_prob=0.2,
        num_classes=int(num_classes),
        last_channel=None,
    )


# ================= EfficientNetV2 1024 非均匀缩放参数 / V2 non-uniform scaling constants =================
# EN: ================= EfficientNetV2 1024 non- parameters / V2 non-uniform scaling constants =================.
# 说明：
# EN: Notes:
# 1. V2 论文指出大分辨率会显著拖慢训练，且等比例放大所有 stage 并非最优；
# EN: V2 large will train, large stage and non- most;
# 2. 因此 1024 版本采用官方 V2-XL 的非均匀 stage 模板：它相对 V2-L 主要增加后段深度与宽度；
# EN: therefore 1024 use V2-XL non- stage: for V2-L need after deep degree and degree;
# 3. 输入分辨率提升到 1024，但不在 V2-XL 之上继续做全局放大，避免训练队列出现过度庞大的模型。
# EN: to 1024, not in V2-XL on do full large, avoidtrain column degree large.
EFFICIENTNET_V2_XL_REFERENCE_RESOLUTION = 512
EFFICIENTNET_V2_1024_WIDTH_MULT = 1.0
EFFICIENTNET_V2_1024_DEPTH_MULT_BY_STAGE = (1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00)


def _scaled_channels(channels: int, width_mult: float) -> int:
    """Scale channel width and keep torchvision EfficientNet's 8-channel divisibility."""
    return int(MBConvConfig.adjust_channels(int(channels), float(width_mult)))


def _scaled_layers(num_layers: int, depth_mult: float) -> int:
    """Scale repeat count with ceil, matching EfficientNet depth scaling behavior."""
    return int(math.ceil(int(num_layers) * float(depth_mult)))


def _scaled_fused_conf(
    expand_ratio: float,
    kernel: int,
    stride: int,
    input_channels: int,
    out_channels: int,
    num_layers: int,
    width_mult: float,
    depth_mult: float,
) -> FusedMBConvConfig:
    """Create a scaled Fused-MBConv stage for early EfficientNetV2 layers."""
    return FusedMBConvConfig(
        expand_ratio,
        kernel,
        stride,
        _scaled_channels(input_channels, width_mult),
        _scaled_channels(out_channels, width_mult),
        _scaled_layers(num_layers, depth_mult),
    )


def _scaled_mbconv_conf(
    expand_ratio: float,
    kernel: int,
    stride: int,
    input_channels: int,
    out_channels: int,
    num_layers: int,
    width_mult: float,
    depth_mult: float,
) -> MBConvConfig:
    """Create a scaled MBConv+SE stage for later EfficientNetV2 layers."""
    return MBConvConfig(
        expand_ratio,
        kernel,
        stride,
        _scaled_channels(input_channels, width_mult),
        _scaled_channels(out_channels, width_mult),
        _scaled_layers(num_layers, depth_mult),
    )


def build_efficientnet_v2_1024(num_classes: int = 1000) -> EfficientNet:
    """
    Build a custom EfficientNetV2 backbone for 1024x1024 input.

    说明：
    English: :
    - 前三段继续使用 Fused-MBConv，保持 V2 对早期层训练速度的优化；
    English: - Fused-MBConv, V2 training;
    - 后四段使用 MBConv+SE，并采用官方 V2-XL 相对 V2-L 更深更宽的非均匀 stage 配置；
    English: - MBConv+SE, V2-XL V2-L stage configuration;
    - 该结构是 1024 输入实验用外推模型，不加载 ImageNet 预训练权重；
    English: - 1024 Inputmodel, load ImageNet training;
    - 结构保留 EfficientNet 默认 BatchNorm；训练时由外层冻结 BN running statistics 处理小批次稳定性；
    English: - EfficientNet default BatchNorm; training BN running statistics ;
    - 最近修改时间：2026-05-22。
    English: - Last modified: 2026-05-22.
    """
    width_mult = float(EFFICIENTNET_V2_1024_WIDTH_MULT)
    depth_mults = EFFICIENTNET_V2_1024_DEPTH_MULT_BY_STAGE
    inverted_residual_setting: List[Union[MBConvConfig, FusedMBConvConfig]] = [
        _scaled_fused_conf(1, 3, 1, 32, 32, 4, width_mult, depth_mults[0]),
        _scaled_fused_conf(4, 3, 2, 32, 64, 8, width_mult, depth_mults[1]),
        _scaled_fused_conf(4, 3, 2, 64, 96, 8, width_mult, depth_mults[2]),
        _scaled_mbconv_conf(4, 3, 2, 96, 192, 16, width_mult, depth_mults[3]),
        _scaled_mbconv_conf(6, 3, 1, 192, 256, 24, width_mult, depth_mults[4]),
        _scaled_mbconv_conf(6, 3, 2, 256, 512, 32, width_mult, depth_mults[5]),
        _scaled_mbconv_conf(6, 3, 1, 512, 640, 8, width_mult, depth_mults[6]),
    ]
    last_channel = _scaled_channels(1280, width_mult)
    return EfficientNet(
        inverted_residual_setting,
        dropout=0.5,
        stochastic_depth_prob=0.2,
        num_classes=int(num_classes),
        last_channel=last_channel,
    )


