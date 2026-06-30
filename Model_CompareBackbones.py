# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
AllBackbones Compare 多模态模型构建代码。
AllBackbones comparison multimodal model builders.

逻辑说明 / Logic
English: Logic.
----------------
1. 本文件只负责 Compare baseline 的模型结构，不承载训练循环、数据加载、checkpoint 或指标导出。
English: 1. file Compare baseline model, training, load, checkpoint metricexport.
2. Compare baseline 不是纯图像模型，而是 image backbone + HyperVISNIR 分支 + NIR 分支 + fusion head。
English: 2. Compare baseline imagemodel, image backbone + HyperVISNIR + NIR + fusion head.
3. HyperVISNIR / NIR 分支直接复用 `Model_MFPCHFNet.py` 中的实现，保证其他模态分支与 MFPC-HFNet 当前活跃代码对齐。
English: NIR 分支直接复用 `Model_MFPCHFNet.py` 中的实现，保证其他模态分支与 MFPC-HFNet 当前活跃代码对齐.
4. 图像 backbone 只输出图像特征，再经 `image_proj` 映射到共享图像嵌入维度；不同 baseline 的差异集中在图像 backbone。
English: 4. image backbone Outputimage, `image_proj` image; baseline image backbone.
5. Compare 模型向 Train 层提供通用 optimizer 参数分组语义，训练层据此执行分组学习率和冻结预热。
English: 5. Compare model Train general optimizer parameter groups, traininglearning rate.
6. 最近修改时间：2026-05-29；作者：ljy。
English: 6. Last modified: 2026-05-29; Author: ljy.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model_MFPCHFNet import HyperBranch, NIRBranch


BACKBONE_ALIAS = {
    "baseline_resnet50": "resnet50",
    "baseline_resnext101_64x4d": "resnext101_64x4d",
    "baseline_efficientnet_b0": "efficientnet_b0",
    "baseline_efficientnet_b4": "efficientnet_b4",
    "baseline_efficientnet_b7": "efficientnet_b7",
    "baseline_efficientnet_v1_1024": "efficientnet_v1_1024",
    "baseline_efficientnet_v2_b0": "efficientnet_v2_b0",
    "baseline_efficientnet_v2_s": "efficientnet_v2_s",
    "baseline_efficientnet_v2_l": "efficientnet_v2_l",
    "baseline_efficientnet_v2_1024": "efficientnet_v2_1024",
    "baseline_vit_l_16": "vit_l_16",
    "baseline_swin_transformer_v2_small": "swin_transformer_v2_small",
    "baseline_swin_transformer_v2_base": "swin_transformer_v2_base",
    "baseline_convnext_small": "convnext_small",
    "baseline_convnext_large": "convnext_large",
    "baseline_convnext_xlarge": "convnext_xlarge",
    "baseline_mobilenetv4_conv_small": "mobilenetv4_conv_small",
    "baseline_mobilenetv4_conv_medium": "mobilenetv4_conv_medium",
    "baseline_mobilenetv4_conv_large": "mobilenetv4_conv_large",
}


def normalize_backbone_name(backbone_name: str) -> str:
    """
    规范化菜单传入的 backbone 名称。
    English: normalizemenu backbone name.

    输入:
    English: Input:
        backbone_name: `Menu_Compare_AllBackbones.py` 中声明的 backbone_name。
        English: backbone_name: `Menu_Compare_AllBackbones.py` backbone_name.
    输出:
    English: Output:
        本文件内部使用的短名称。
        English: filename.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    name = str(backbone_name).strip()
    if not name:
        raise ValueError("Compare 模型必须提供 backbone_name。")
    return BACKBONE_ALIAS.get(name, name)


def replace_first_conv_in_channels(module: nn.Module, in_channels: int) -> None:
    """
    将图像 backbone 的第一个 RGB Conv2d 改为本项目 8 通道输入。
    English: image backbone RGB Conv2d 8 Input.

    物理/工程意义:
    English: /:
    - 本项目土壤图像为 8 通道，不使用 ImageNet RGB 预训练权重；
    English: - image 8 , ImageNet RGB training;
    - 因此第一个卷积只需要保持原输出通道、kernel、stride、padding 等结构参数，并把 in_channels 改为 8；
    English: - Output, kernel, stride, padding parameter, in_channels 8;
    - 只替换遇到的第一个 Conv2d，避免误改中间卷积层。
    English: - Conv2d, avoid.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    target_channels = int(in_channels)
    for child_name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and int(child.in_channels) != target_channels:
            replacement = nn.Conv2d(
                in_channels=target_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            setattr(module, child_name, replacement)
            return
        replace_first_conv_in_channels(child, target_channels)
        first_conv = next((m for m in child.modules() if isinstance(m, nn.Conv2d) and int(m.in_channels) == target_channels), None)
        if first_conv is not None:
            return


def disable_stochastic_regularization(module: nn.Module) -> None:
    """
    可选关闭 baseline backbone 内部 dropout / stochastic depth。
    English: stochastic depth.

    注意:
    English: :
    - 默认不调用该函数，保留菜单中的 `disable_dropout_droppath=False` 行为；
    English: - default, menu `disable_dropout_droppath=False` ;
    - 该函数只在菜单显式要求关闭时使用，不影响 Hyper/NIR/fusion 分支设计。
    English: - menuexplicit, Hyper/NIR/fusion .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout):
            submodule.p = 0.0
        if hasattr(submodule, "stochastic_depth_prob"):
            submodule.stochastic_depth_prob = 0.0
        if hasattr(submodule, "drop_prob"):
            submodule.drop_prob = 0.0


def _replace_classifier_with_identity(model: nn.Module, classifier_name: str) -> int:
    """
    将常见 torchvision 分类头替换为 Identity 并返回特征维度。
    English: torchvision Identity return.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    classifier = getattr(model, classifier_name)
    if isinstance(classifier, nn.Linear):
        feature_dim = int(classifier.in_features)
    elif isinstance(classifier, nn.Sequential):
        linear_layers = [layer for layer in classifier.modules() if isinstance(layer, nn.Linear)]
        if not linear_layers:
            raise ValueError(f"{classifier_name} 中未找到 Linear 分类层，无法判断特征维度。")
        feature_dim = int(linear_layers[-1].in_features)
    else:
        raise ValueError(f"暂不支持的分类头类型: {type(classifier).__name__}")
    setattr(model, classifier_name, nn.Identity())
    return feature_dim


def build_torchvision_feature_backbone(backbone_name: str, image_channels: int, expected_image_hw: Sequence[int]) -> tuple[nn.Module, int]:
    """
    构建 torchvision / 项目自定义 EfficientNet 图像特征主干。
    English: 项目自定义 EfficientNet 图像特征主干.

    输入:
    English: Input:
        backbone_name: 规范化后的 backbone 短名称。
        English: backbone_name: normalize backbone name.
        image_channels: 输入图像通道数，当前为 8。
        English: image_channels: Inputimage, current 8.
        expected_image_hw: 菜单声明的输入图像尺寸，用于 ViT 等固定位置编码模型。
        English: expected_image_hw: menuInputimage, ViT model.
    输出:
    English: Output:
        (backbone, feature_dim)，backbone 输出 [B, feature_dim] 或可展平为该维度。
        English: (backbone, feature_dim), backbone Output [B, feature_dim] .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    from torchvision import models

    if backbone_name == "resnet50":
        model = models.resnet50(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "fc")
    elif backbone_name == "resnext101_64x4d":
        model = models.resnext101_64x4d(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "fc")
    elif backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    elif backbone_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    elif backbone_name == "efficientnet_b7":
        model = models.efficientnet_b7(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    elif backbone_name == "efficientnet_v2_b0":
        model = models.efficientnet_v2_s(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    elif backbone_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    elif backbone_name == "efficientnet_v2_l":
        model = models.efficientnet_v2_l(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    elif backbone_name == "efficientnet_v1_1024":
        from Model_EfficientNet1024Backbones import build_efficientnet_v1_1024

        model = build_efficientnet_v1_1024(num_classes=1000)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    elif backbone_name == "efficientnet_v2_1024":
        from Model_EfficientNet1024Backbones import build_efficientnet_v2_1024

        model = build_efficientnet_v2_1024(num_classes=1000)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    elif backbone_name == "vit_l_16":
        side = int(expected_image_hw[0])
        model = models.vit_l_16(weights=None, image_size=side)
        feature_dim = _replace_classifier_with_identity(model, "heads")
    elif backbone_name == "swin_transformer_v2_small":
        model = models.swin_v2_s(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "head")
    elif backbone_name == "swin_transformer_v2_base":
        model = models.swin_v2_b(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "head")
    elif backbone_name == "convnext_small":
        model = models.convnext_small(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    elif backbone_name == "convnext_large":
        model = models.convnext_large(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    elif backbone_name == "convnext_xlarge":
        model = models.convnext_large(weights=None)
        feature_dim = _replace_classifier_with_identity(model, "classifier")
    else:
        raise ValueError(f"未知 torchvision Compare backbone: {backbone_name}")

    replace_first_conv_in_channels(model, int(image_channels))
    return model, int(feature_dim)


def build_timm_feature_backbone(backbone_name: str, image_channels: int) -> tuple[nn.Module, int]:
    """
    构建 timm 图像特征主干，当前用于 MobileNetV4。
    English: build timm image, current MobileNetV4.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    import timm

    candidates = {
        "mobilenetv4_conv_small": ("mobilenetv4_conv_small.e2400_r224_in1k", "mobilenetv4_conv_small"),
        "mobilenetv4_conv_medium": ("mobilenetv4_conv_medium.e500_r256_in1k", "mobilenetv4_conv_medium"),
        "mobilenetv4_conv_large": ("mobilenetv4_conv_large.e600_r384_in1k", "mobilenetv4_conv_large"),
    }[backbone_name]
    last_error: Optional[Exception] = None
    for model_name in candidates:
        try:
            model = timm.create_model(
                model_name,
                pretrained=False,
                in_chans=int(image_channels),
                num_classes=0,
                global_pool="avg",
            )
            return model, int(model.num_features)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法构建 timm backbone {backbone_name}: {last_error}")


class ImageBackboneFeatureExtractor(nn.Module):
    """
    Compare baseline 图像主干特征提取器。
    English: Compare baseline image.

    输入:
    English: Input:
        image: [B, 8, H, W] 多通道土壤图像。
        English: image: [B, 8, H, W] image.
    输出:
    English: Output:
        [B, feature_dim] 图像 backbone 特征。
        English: [B, feature_dim] image backbone .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    def __init__(
        self,
        backbone_name: str,
        image_channels: int = 8,
        expected_image_hw: Sequence[int] = (256, 256),
        disable_dropout_droppath: bool = False,
    ):
        super().__init__()
        self.backbone_name = normalize_backbone_name(backbone_name)
        self.expected_image_hw = (int(expected_image_hw[0]), int(expected_image_hw[1]))

        if self.backbone_name.startswith("mobilenetv4_"):
            self.backbone, self.feature_dim = build_timm_feature_backbone(self.backbone_name, int(image_channels))
        else:
            self.backbone, self.feature_dim = build_torchvision_feature_backbone(
                self.backbone_name,
                int(image_channels),
                self.expected_image_hw,
            )

        if bool(disable_dropout_droppath):
            disable_stochastic_regularization(self.backbone)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        提取并展平图像 backbone 特征。
        English: image backbone .
        """

        if image.shape[-2:] != self.expected_image_hw:
            image = F.interpolate(image, size=self.expected_image_hw, mode="bilinear", align_corners=False)
        feat = self.backbone(image)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return feat


class CompareBackboneMultiModalRegressor(nn.Module):
    """
    AllBackbones Compare 多模态回归模型。
    English: AllBackbones Compare model.

    结构:
        image -> image backbone -> image_proj -> image_embed_dim
        hyper -> HyperBranch -> hyper_embed_dim
        nir   -> NIRBranch   -> nir_embed_dim
        concat -> fusion_head -> output_dim

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    def __init__(
        self,
        backbone_name: str,
        output_dim: int,
        feature_dim_hyper: int = 681,
        feature_dim_nir: int = 5,
        image_channels: int = 8,
        expected_image_hw: Sequence[int] = (256, 256),
        image_embed_dim: int = 24,
        hyper_embed_dim: int = 32,
        nir_embed_dim: int = 16,
        fusion_hidden_dim: int = 48,
        image_head_dropout: float = 0.1,
        fusion_dropout: float = 0.15,
        disable_dropout_droppath: bool = False,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        if self.output_dim not in (1, 2):
            raise ValueError(f"output_dim 仅支持 1 或 2，当前为 {self.output_dim}。")

        self.image_branch = ImageBackboneFeatureExtractor(
            backbone_name=backbone_name,
            image_channels=image_channels,
            expected_image_hw=expected_image_hw,
            disable_dropout_droppath=disable_dropout_droppath,
        )
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_branch.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(float(image_head_dropout)),
            nn.Linear(128, int(image_embed_dim)),
            nn.ReLU(inplace=True),
        )
        self.hyper_branch = HyperBranch(in_dim=int(feature_dim_hyper), embed_dim=int(hyper_embed_dim))
        self.nir_branch = NIRBranch(in_dim=int(feature_dim_nir), embed_dim=int(nir_embed_dim))

        fusion_in_dim = int(image_embed_dim) + int(hyper_embed_dim) + int(nir_embed_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, int(fusion_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(fusion_hidden_dim)),
            nn.Dropout(float(fusion_dropout)),
            nn.Linear(int(fusion_hidden_dim), 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.output_dim),
        )

    def get_structure_summary(self) -> dict:
        """
        返回 Compare 多模态模型结构摘要，供日志或诊断使用。
        English: return Compare model, .
        """

        return {
            "model_family": "compare_backbone",
            "backbone_name": self.image_branch.backbone_name,
            "image_feature_dim": int(self.image_branch.feature_dim),
            "image_branch": "torchvision_or_timm_backbone",
            "hyper_branch": "Model_MFPCHFNet.HyperBranch",
            "nir_branch": "Model_MFPCHFNet.NIRBranch",
            "fusion_head": "Linear-ReLU-LayerNorm-Dropout-Linear-ReLU-Linear",
        }

    def get_optimizer_parameter_groups(self) -> list[dict]:
        """
        返回 Compare 微调策略使用的参数分组。
        English: return Compare parameter groups.

        逻辑 / Logic:
        English: Logic:.
        1. `backbone` 组仅包含图像 backbone 原始主干参数，对应历史 SwinV2 微调中的低学习率/冻结预热对象；
        English: 1. `backbone` image backbone parameter, SwinV2 learning rate/;
        2. `head` 组包含 image projection、HyperVISNIR、NIR 和 fusion head，对应较高学习率的任务适配层；
        English: 2. `head` image projection, HyperVISNIR, NIR fusion head, learning rate;
        3. 本函数只声明模型内部参数语义，不创建 optimizer，也不决定具体学习率数值。
        English: 3. modelparameter, create optimizer, learning rate.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """

        return [
            {
                "name": "head",
                "lr_role": "head",
                "params": [
                    *self.image_proj.parameters(),
                    *self.hyper_branch.parameters(),
                    *self.nir_branch.parameters(),
                    *self.fusion_head.parameters(),
                ],
            },
            {
                "name": "backbone",
                "lr_role": "backbone",
                "params": self.image_branch.backbone.parameters(),
            },
        ]

    def forward(self, hyper: torch.Tensor, nir: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        执行 Compare baseline 多模态回归前向传播。
        English: Compare baseline .
        """

        img_feat = self.image_proj(self.image_branch(image))
        hyper_feat = self.hyper_branch(hyper)
        nir_feat = self.nir_branch(nir)
        fused = torch.cat([img_feat, hyper_feat, nir_feat], dim=1)
        out = self.fusion_head(fused)
        if self.output_dim == 1:
            return out.view(-1)
        return out


def build_compare_backbone_model(
    backbone_name: str,
    output_dim: int,
    feature_dim_hyper: int = 681,
    feature_dim_nir: int = 5,
    image_channels: int = 8,
    expected_image_hw: Sequence[int] = (256, 256),
    image_embed_dim: int = 24,
    hyper_embed_dim: int = 32,
    nir_embed_dim: int = 16,
    fusion_hidden_dim: int = 48,
    disable_dropout_droppath: bool = False,
) -> CompareBackboneMultiModalRegressor:
    """
    Compare baseline 模型工厂函数。
    English: Compare baseline model.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    return CompareBackboneMultiModalRegressor(
        backbone_name=backbone_name,
        output_dim=output_dim,
        feature_dim_hyper=feature_dim_hyper,
        feature_dim_nir=feature_dim_nir,
        image_channels=image_channels,
        expected_image_hw=expected_image_hw,
        image_embed_dim=image_embed_dim,
        hyper_embed_dim=hyper_embed_dim,
        nir_embed_dim=nir_embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        disable_dropout_droppath=disable_dropout_droppath,
    )


__all__ = [
    "CompareBackboneMultiModalRegressor",
    "ImageBackboneFeatureExtractor",
    "build_compare_backbone_model",
    "normalize_backbone_name",
]
