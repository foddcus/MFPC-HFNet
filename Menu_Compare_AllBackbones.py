# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
AllBackbones 对比训练菜单。
AllBackbones comparison training menu.

逻辑说明 / Logic
English: Logic.
----------------
1. 本文件只维护 baseline/backbone 模型清单、从小模型到大模型的执行顺序、显示名、输入分辨率和菜单级微调标签。
English: 1. file baseline/backbone model, modelmodel, , Inputmenulabel.
2. 本文件不承载训练循环、数据加载、复杂度导出或结果汇总；这些统一收敛到 `Train_core.py` 与 `Metrics_*.py`。
English: 2. filetraining, load, exportresult; `Train_core.py` `Metrics_*.py`.
3. Compare 共享嵌入/融合维度属于模型结构字段，只在菜单 Config 中声明，不由 Train_main.py 入口面板设计。
English: 3. Compare /modelfield, menu Config , Train_main.py .
4. SwinV2 Base 的冻结预热和分组学习率只作为菜单微调标签传给训练引擎，不在菜单中执行；训练引擎不得自行补写未在菜单声明的特殊参数。
English: 4. SwinV2 Base learning ratemenulabeltraining engine, menu; training enginemenuparameter.
5. V2 禁止出现多个训练执行主体；本菜单不导入任何此类模块。
English: 5. V2 training; menu.

最近修改时间 / Last modified: 2026-05-30
English: Last modified: 2026-05-30.
作者 / Author: ljy
English: Author: ljy.
"""

from __future__ import annotations

from Train_core import CommonTrainConfig, ExperimentMenu, ModelSpec, TrainingConfig, run_training_menu


class Config(CommonTrainConfig):
    """
    AllBackbones 对比菜单默认训练参数。
    English: AllBackbones menudefaulttrainingparameter.

    最近修改时间 / Last modified: 2026-05-30
    English: Last modified: 2026-05-30.
    作者 / Author: ljy
    English: Author: ljy.
    维护记录 / Maintenance:
    English: Maintenance:.
    - 2026-05-30；作者：ljy。接管 Compare 共享嵌入/融合维度，Train_main.py 不再维护这些模型结构字段。
    English: - 2026-05-30; Author: ljy. Compare /, Train_main.py modelfield.
    """

    TRAIN_MODEL_PRESET = "all"
    TRAIN_MODEL_NAMES: list[str] = []
    RESUME_TRAINING = True
    RESUME_SAVE_DIR = None
    MAX_EPOCHS = 1000
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    SHARED_IMAGE_EMBED_DIM = 24  # 共享图像分支嵌入维度。 / EN: image.
    SHARED_HYPER_EMBED_DIM = 32  # 共享 Hyper 分支嵌入维度。 / EN: Hyper.
    SHARED_NIR_EMBED_DIM = 16  # 共享 NIR 分支嵌入维度。 / EN: NIR.
    SHARED_FUSION_HIDDEN_DIM = 48  # 共享融合头隐藏层维度。 / EN: fusion.


COMPARE_CONFIG = TrainingConfig(
    name="compare_all_backbones",
    active_entrypoint="Menu_Compare_AllBackbones.py",
    manifest_filename="compare_run_manifest.json",
    summary_prefix="compare_all_models",
)


def _size(tag: str) -> tuple[int, int]:
    """
    把 256x256 形式的分辨率标签转为二元组。
    English: 256x256 label.
    """

    side_a, side_b = tag.lower().split("x", 1)
    return int(side_a), int(side_b)


def _compare_spec(name: str, display_name: str, backbone_name: str, resolution: str, **kwargs) -> ModelSpec:
    """
    构造对比模型菜单条目。
    English: modelmenu.
    """

    return ModelSpec(
        name=name,
        display_name=display_name,
        model_family="compare_backbone",
        backbone_name=backbone_name,
        resolution=resolution,
        image_size=_size(resolution),
        batch_size=32,
        **kwargs,
    )


# ================= AllBackbones 对比菜单 =================
# EN: ================= AllBackbones for single =================.
# 逻辑 / Logic:
# EN: logic / Logic:
# 1. 所有条目的 batch_size 统一为 32，作为模型训练清单规格，不表达硬件资源策略；
# EN: batch_size as 32, as training list, not table source;
# 2. SwinV2 Base 使用冻结预热 + 分组学习率标签，供 Train_core 后续统一解释；
# EN: SwinV2 Base use result + grouped learning rateslabel, Train_core later;
# 3. 最近修改时间：2026-05-29；作者：ljy。
# EN: Last modified: 2026-05-29; Author: ljy.
COMPARE_ALL_MODEL_SPECS = [
    _compare_spec("ResNet50_small", "ResNet-50", "baseline_resnet50", "256x256"),
    _compare_spec("ResNeXt101_64X4D_small", "ResNeXt101-64X4D", "baseline_resnext101_64x4d", "256x256"),
    _compare_spec("ResNeXt101_64X4D_512x512", "ResNeXt101-64X4D", "baseline_resnext101_64x4d", "512x512"),
    _compare_spec("EfficientNet_B0_224x224", "EfficientNet-B0", "baseline_efficientnet_b0", "224x224"),
    _compare_spec("EfficientNet_B4_380x380", "EfficientNet-B4", "baseline_efficientnet_b4", "380x380"),
    _compare_spec("EfficientNet_B7_600x600", "EfficientNet-B7", "baseline_efficientnet_b7", "600x600"),
    _compare_spec("EfficientNet_1024x1024", "EfficientNet-1024", "baseline_efficientnet_v1_1024", "1024x1024"),
    _compare_spec("EfficientNetV2_B0_224x224", "EfficientNetV2-B0", "baseline_efficientnet_v2_b0", "224x224"),
    _compare_spec("EfficientNetV2_S_384x384", "EfficientNetV2-S", "baseline_efficientnet_v2_s", "384x384"),
    _compare_spec("EfficientNetV2_L_480x480", "EfficientNetV2-L", "baseline_efficientnet_v2_l", "480x480"),
    _compare_spec("EfficientNetV2_1024x1024", "EfficientNetV2-1024", "baseline_efficientnet_v2_1024", "1024x1024"),
    _compare_spec("ViT_L_16_small", "ViT-L-16", "baseline_vit_l_16", "256x256"),
    _compare_spec("SwinTransformerV2_Small_256x256", "Swin Transformer V2 Small", "baseline_swin_transformer_v2_small", "256x256"),
    _compare_spec(
        "SwinTransformerV2_Base_384x384",
        "Swin Transformer V2 Base",
        "baseline_swin_transformer_v2_base",
        "384x384",
        optimizer_policy="freeze_then_layerwise",
        backbone_lr=1e-5,
        head_lr=1e-4,
        freeze_backbone_epochs=30,
        disable_dropout_droppath=False,
    ),
    _compare_spec("ConvNeXt_Small_small", "ConvNeXt-Small", "baseline_convnext_small", "256x256"),
    _compare_spec("ConvNeXt_Large_384x384", "ConvNeXt-Large", "baseline_convnext_large", "384x384"),
    _compare_spec("ConvNeXt_XLarge_512x512", "ConvNeXt-XLarge", "baseline_convnext_xlarge", "512x512"),
    _compare_spec("MobileNetV4_Conv_Small_256x256", "MobileNetV4-Conv-Small", "baseline_mobilenetv4_conv_small", "256x256"),
    _compare_spec("MobileNetV4_Conv_Medium_320x320", "MobileNetV4-Conv-Medium", "baseline_mobilenetv4_conv_medium", "320x320"),
    _compare_spec("MobileNetV4_Conv_Large_448x448", "MobileNetV4-Conv-Large", "baseline_mobilenetv4_conv_large", "448x448"),
]


COMPARE_MENU = ExperimentMenu(
    config=COMPARE_CONFIG,
    config_cls=Config,
    all_model_specs=COMPARE_ALL_MODEL_SPECS,
    preset_groups={
        "classic_only": [
            "ResNet50_small",
            "ResNeXt101_64X4D_small",
            "ResNeXt101_64X4D_512x512",
            "ViT_L_16_small",
            "SwinTransformerV2_Small_256x256",
            "SwinTransformerV2_Base_384x384",
            "ConvNeXt_Small_small",
            "ConvNeXt_Large_384x384",
            "ConvNeXt_XLarge_512x512",
            "MobileNetV4_Conv_Small_256x256",
            "MobileNetV4_Conv_Medium_320x320",
            "MobileNetV4_Conv_Large_448x448",
        ],
        "efficientnet_only": [
            "EfficientNet_B0_224x224",
            "EfficientNet_B4_380x380",
            "EfficientNet_B7_600x600",
            "EfficientNet_1024x1024",
            "EfficientNetV2_B0_224x224",
            "EfficientNetV2_S_384x384",
            "EfficientNetV2_L_480x480",
            "EfficientNetV2_1024x1024",
        ],
    },
)


def get_compare_all_backbones_menu() -> ExperimentMenu:
    """
    返回 AllBackbones 对比训练菜单。
    English: return AllBackbones trainingmenu.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    COMPARE_MENU.sync_to_engine()
    return COMPARE_MENU


_MENU = get_compare_all_backbones_menu()
MODEL_SPECS = _MENU.MODEL_SPECS


def select_model_specs() -> list[ModelSpec]:
    """
    按当前 Config 选择本轮训练模型清单。
    English: current Config selecttrainingmodel.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    _MENU.sync_to_engine()
    return _MENU.MODEL_SPECS


def main():
    """
    运行 AllBackbones 对比训练菜单。
    English: AllBackbones trainingmenu.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    return run_training_menu(_MENU)


__all__ = [
    "COMPARE_ALL_MODEL_SPECS",
    "COMPARE_CONFIG",
    "COMPARE_MENU",
    "Config",
    "MODEL_SPECS",
    "ModelSpec",
    "get_compare_all_backbones_menu",
    "main",
    "select_model_specs",
]


if __name__ == "__main__":
    main()
