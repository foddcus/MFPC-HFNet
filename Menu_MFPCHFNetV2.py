# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
MFPC-HFNet 结构训练菜单。
MFPC-HFNet structure training menu.

逻辑说明 / Logic
English: Logic.
----------------
1. 本文件只承担“菜单”职责：声明本次可训练模型、从小模型到大模型的执行顺序、结构标签、输入尺寸、batch size 和菜单默认微调设置。
English: 1. file“menu”: trainingmodel, modelmodel, label, Input, batch size menudefault.
2. 本文件不承载训练循环、数据加载、结果导出或断点续训实现；这些均由 `Train_core.py` 统一处理。
English: 2. filetraining, load, resultexport; `Train_core.py` .
3. 模型具体构建代码属于 `Model_MFPCHFNet.py`，菜单负责把后续调用模型代码所需的规格和特殊参数传递清楚。
English: 3. modelbuild `Model_MFPCHFNet.py`, menumodelparameter.
4. MFPC-HFNet 共享嵌入维度、token、attention、PCA 冻结和 dropout 等结构超参数只在本菜单层显式声明，Train_main.py 不再设计这些字段。
English: 4. MFPC-HFNet , token, attention, PCA dropout parametermenuexplicit, Train_main.py field.
5. Transformer 融合/合并层低 LR 只在菜单声明策略意图，由模型层参数分组和 Train_optimizer.py 执行。
English: 5. Transformer / LR menu, modelparameter groups Train_optimizer.py .
6. V2 禁止出现多个训练执行主体；本菜单不导入任何此类模块。
English: 6. V2 training; menu.

最近修改时间 / Last modified: 2026-05-30
English: Last modified: 2026-05-30.
作者 / Author: ljy
English: Author: ljy.
"""

from __future__ import annotations

from Train_core import CommonTrainConfig, ExperimentMenu, ModelSpec, TrainingConfig, run_training_menu


class Config(CommonTrainConfig):
    """
    MFPC-HFNet 菜单默认训练参数。
    English: MFPC-HFNet menudefaulttrainingparameter.

    最近修改时间 / Last modified: 2026-05-30
    English: Last modified: 2026-05-30.
    作者 / Author: ljy
    English: Author: ljy.
    维护记录 / Maintenance:
    English: Maintenance:.
    - 2026-05-30；作者：ljy。接管 MFPC-HFNet 共享嵌入维度和结构超参数，Train_main.py 不再维护这些模型结构字段。
    English: - 2026-05-30; Author: ljy. MFPC-HFNet parameter, Train_main.py modelfield.
    - 2026-05-30；作者：ljy。显式声明 MFPC-HFNet 各结构输入尺寸，Train_core.py 不再提供隐藏结构尺寸默认值。
    English: - 2026-05-30; Author: ljy.explicit MFPC-HFNet Input, Train_core.py default.
    """

    TRAIN_MODEL_PRESET = "all"
    TRAIN_MODEL_NAMES: list[str] = []
    RESUME_TRAINING = False
    RESUME_SAVE_DIR = None
    MAX_EPOCHS = 1000
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 3e-4
    SHARED_IMAGE_EMBED_DIM = 24  # 共享图像分支嵌入维度。 / EN: image.
    SHARED_HYPER_EMBED_DIM = 32  # 共享 Hyper 分支嵌入维度。 / EN: Hyper.
    SHARED_NIR_EMBED_DIM = 16  # 共享 NIR 分支嵌入维度。 / EN: NIR.
    SHARED_FUSION_HIDDEN_DIM = 48  # 共享融合头隐藏层维度。 / EN: fusion.
    INPUT_FEATURE_VECTOR_RATIO = 0.045  # 旧 PCA 先验缺失时的特征向量比例 fallback。 / EN: old PCA when fallback.
    ALLOCATION_SOURCE = "eigvals"  # PCASE 通道分配依据。 / EN: PCASE.
    FREEZE_PCA = True  # 是否冻结 PCA 投影。 / EN: is PCA.
    MFPCHF_TOKEN_COMPRESSION_RATIO = 8.0  # MFPC-HFNet token 压缩倍率。 / EN: MFPC-HFNet token compression ratio.
    MFPCHF_TOKEN_DIM_MIN = 96  # MFPC-HFNet token 维度下限。 / EN: MFPC-HFNet token lower bound.
    MFPCHF_TOKEN_DIM_ROUND_MULTIPLE = 16  # MFPC-HFNet token 维度取整倍数。 / EN: MFPC-HFNet token.
    MFPCHF_LD_ATTN_DIM = 64  # LD Encoder 注意力维度。 / EN: LD Encoder.
    MFPCHF_LD_HEADS = 2  # LD Encoder 注意力头数。 / EN: LD Encoder.
    MFPCHF_CPE_ATTN_DIM = 64  # CrossPatchEncoder 注意力维度。 / EN: CrossPatchEncoder.
    MFPCHF_CPE_HEADS = 2  # CrossPatchEncoder 注意力头数。 / EN: CrossPatchEncoder.
    MFPCHF_DROPOUT = 0.0  # MFPC-HFNet 全局注意力路径 dropout；当前不再设置分频 dropout。 / EN: MFPC-HFNet full path dropout; currentno longer dropout.
    FULL_IMAGE_SIZE = (1024, 1024)  # Full / 3HF 结构输入尺寸。 / EN: Full / 3HF input size.
    H2H3LOW_IMAGE_SIZE = (512, 512)  # H2H3Low / 2HF 结构输入尺寸。 / EN: H2H3Low / 2HF input size.
    H3LOW_IMAGE_SIZE = (256, 256)  # H3Low / 1HF 结构输入尺寸。 / EN: H3Low / 1HF input size.
    LOWONLY_IMAGE_SIZE = (128, 128)  # LowOnly 结构输入尺寸。 / EN: LowOnly input size.


MFPCHFNETV2_CONFIG = TrainingConfig(
    name="mfpchfnetv2",
    active_entrypoint="Menu_MFPCHFNetV2.py",
    manifest_filename="mfpchfnetv2_unified_manifest.json",
    summary_prefix="mfpchfnetv2_unified",
)


MFPCHF_TRANSFORMER_FUSION_OPTIMIZER_EXTRA = {
    "optimizer_role_lr_scales": {
        "default": 1.0,
        "transformer_fusion": 0.1,
    },
}


# ================= MFPC-HFNet 结构菜单 =================
# EN: ================= MFPC-HFNet result single =================.
# 逻辑 / Logic:
# EN: logic / Logic:
# 1. Full 对应 high1 + high2 + high3 + low，输入 1024x1024；
# EN: Full for should high1 + high2 + high3 + low, 1024x1024;
# 2. H2H3Low 对应 512 分辨率结构分支，不是把 Full 改成 512；
# EN: H2H3Low for should 512 result, not is Full change 512;
# 3. batch_size 是模型训练清单规格，不表达 GPU auto-batch/OOM 硬件策略；
# EN: batch_size is training list, not table GPU auto-batch/OOM;
# 4. optimizer_policy="layerwise_lr" 表示启用通用角色分组 LR；transformer_fusion 组使用 0.1×基础 LR；
# EN: optimizer_policy="layerwise_lr" meansenablegeneral LR; transformer_fusion use 0.1x LR;
# 5. 最近修改时间：2026-05-30；作者：ljy。
# EN: Last modified: 2026-05-30; Author: ljy.
MFPCHFNETV2_ALL_MODEL_SPECS = [
    ModelSpec(
        name="MFPCHFNetV2_Full",
        display_name="MFPC-HFNet",
        model_family="mfpchfnet",
        structure="high1+high2+high3+low",
        priors_path=Config.PCA_PRIORS_PATH,
        batch_size=32,
        image_size=Config.FULL_IMAGE_SIZE,
        optimizer_policy="layerwise_lr",
        extra=MFPCHF_TRANSFORMER_FUSION_OPTIMIZER_EXTRA,
    ),
    ModelSpec(
        name="MFPCHFNetV2_H2H3Low",
        display_name="MFPC-HFNet-H2H3Low",
        model_family="mfpchfnet",
        structure="high2+high3+low",
        priors_path=Config.PCA_PRIORS_PATH,
        batch_size=32,
        image_size=Config.H2H3LOW_IMAGE_SIZE,
        optimizer_policy="layerwise_lr",
        extra=MFPCHF_TRANSFORMER_FUSION_OPTIMIZER_EXTRA,
    ),
    ModelSpec(
        name="MFPCHFNetV2_H3Low",
        display_name="MFPC-HFNet-H3Low",
        model_family="mfpchfnet",
        structure="high3+low",
        priors_path=Config.PCA_PRIORS_PATH,
        batch_size=32,
        image_size=Config.H3LOW_IMAGE_SIZE,
        optimizer_policy="layerwise_lr",
        extra=MFPCHF_TRANSFORMER_FUSION_OPTIMIZER_EXTRA,
    ),
    ModelSpec(
        name="MFPCHFNetV2_LowOnly",
        display_name="MFPC-HFNet-LowOnly",
        model_family="mfpchfnet",
        structure="low",
        priors_path=Config.PCA_PRIORS_PATH,
        batch_size=32,
        image_size=Config.LOWONLY_IMAGE_SIZE,
        optimizer_policy="layerwise_lr",
        extra=MFPCHF_TRANSFORMER_FUSION_OPTIMIZER_EXTRA,
    ),
]


MFPCHFNETV2_MENU = ExperimentMenu(
    config=MFPCHFNETV2_CONFIG,
    config_cls=Config,
    all_model_specs=MFPCHFNETV2_ALL_MODEL_SPECS,
    preset_groups={
        "full_only": ["MFPCHFNetV2_Full"],
        "ablation_only": ["MFPCHFNetV2_H2H3Low", "MFPCHFNetV2_H3Low", "MFPCHFNetV2_LowOnly"],
    },
)


def get_mfpchfnetv2_menu() -> ExperimentMenu:
    """
    返回 MFPC-HFNet 结构训练菜单。
    English: return MFPC-HFNet trainingmenu.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    MFPCHFNETV2_MENU.sync_to_engine()
    return MFPCHFNETV2_MENU


_MENU = get_mfpchfnetv2_menu()
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
    运行 MFPC-HFNet 训练菜单。
    English: MFPC-HFNet trainingmenu.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    return run_training_menu(_MENU)


__all__ = [
    "Config",
    "MFPCHF_TRANSFORMER_FUSION_OPTIMIZER_EXTRA",
    "MFPCHFNETV2_ALL_MODEL_SPECS",
    "MFPCHFNETV2_CONFIG",
    "MFPCHFNETV2_MENU",
    "MODEL_SPECS",
    "ModelSpec",
    "get_mfpchfnetv2_menu",
    "main",
    "select_model_specs",
]


if __name__ == "__main__":
    main()
