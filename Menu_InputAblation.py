# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
输入端消融训练菜单。
Input-ablation training menu.

逻辑说明 / Logic
English: Logic.
----------------
1. 本文件只声明 6 个待训练输入组合，并按从小模型到大模型的顺序交给主管执行；Full 基线只读导入信息不写入训练模型清单。
English: 1. file 6 trainingInput, modelmodel; Full writetrainingmodel.
2. 训练循环、shared_folds 读取、Full 结果导入和输出汇总均由 `Train_core.py` / `Metrics_*.py` 后续统一实现。
English: `Metrics_*.py` 后续统一实现.
3. 特殊训练参数、模型输入组合、共享嵌入维度和 MFPC-HFNet 结构超参数必须先写在菜单中；训练引擎后续只能按菜单条目调用 `Model_*.py` 构建未训练模型。
English: 3. trainingparameter, modelInput, MFPC-HFNet parametermenu; training enginemenu `Model_*.py` buildtrainingmodel.
4. 菜单层不声明 GPU auto-batch/OOM 等硬件资源策略。
English: 4. menu GPU auto-batch/OOM .
5. V2 禁止出现多个训练执行主体；本菜单不导入任何此类模块。
English: 5. V2 training; menu.

最近修改时间 / Last modified: 2026-05-30
English: Last modified: 2026-05-30.
作者 / Author: ljy
English: Author: ljy.
维护记录 / Maintenance:
English: Maintenance:.
- 2026-05-29；作者：ljy。将输入端消融菜单的 WEIGHT_DECAY 对齐到 Menu_MFPCHFNetV2.py；LEARNING_RATE 原本已同为 1e-4。
English: - 2026-05-29; Author: ljy.Inputmenu WEIGHT_DECAY Menu_MFPCHFNetV2.py; LEARNING_RATE 1e-4.
- 2026-05-29；作者：ljy。将不含图像的 NIR / Hyper / Hyper+NIR 输入组合 batch_size 调整为 256，图像相关组合保持 10。
English: Hyper / Hyper+NIR 输入组合 batch_size 调整为 256，图像相关组合保持 10.
- 2026-05-30；作者：ljy。接管输入端消融使用的共享嵌入维度、MFPC-HFNet 结构超参数和 Full 参考模型名，Train_main.py 不再维护这些字段。
English: - 2026-05-30; Author: ljy.Input, MFPC-HFNet parameter Full model, Train_main.py field.
"""

from __future__ import annotations

from Train_core import CommonTrainConfig, ExperimentMenu, ModelSpec, TrainingConfig, run_training_menu


class Config(CommonTrainConfig):
    """
    输入端消融菜单默认训练参数。
    English: Inputmenudefaulttrainingparameter.

    最近修改时间 / Last modified: 2026-05-30
    English: Last modified: 2026-05-30.
    作者 / Author: ljy
    English: Author: ljy.
    维护记录 / Maintenance:
    English: Maintenance:.
    - 2026-05-29；作者：ljy。基础学习率与 MFPC-HFNetV2 菜单保持 1e-4，权重衰减对齐为 3e-4。
    English: - 2026-05-29; Author: ljy.learning rate MFPC-HFNetV2 menu 1e-4, 3e-4.
    - 2026-05-30；作者：ljy。接管共享嵌入维度、MFPC-HFNet 结构超参数和 Full 参考模型名，保持结构实验参数只在菜单层显式声明。
    English: - 2026-05-30; Author: ljy., MFPC-HFNet parameter Full model, parametermenuexplicit.
    - 2026-05-30；作者：ljy。显式声明输入端消融图像尺寸和图像输入组合结构标签，Train_core.py 不再推断 Full。
    English: - 2026-05-30; Author: ljy.explicitInputimageimageInputlabel, Train_core.py Full.
    """

    TRAIN_MODEL_PRESET = "all"
    TRAIN_MODEL_NAMES: list[str] = []
    RESUME_TRAINING = False
    RESUME_SAVE_DIR = None
    MAX_EPOCHS = 1000
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 3e-4
    LOAD_SHARED_FOLDS_FROM_CSV = True
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
    FULL_IMAGE_SIZE = (1024, 1024)  # 输入端消融沿用 Full 图像分支尺寸。 / EN: input-side ablation use Full image.
    H2H3LOW_IMAGE_SIZE = (512, 512)  # 结构尺寸边界保留给菜单显式声明。 / EN: keep single explicitly declare.
    H3LOW_IMAGE_SIZE = (256, 256)  # 结构尺寸边界保留给菜单显式声明。 / EN: keep single explicitly declare.
    LOWONLY_IMAGE_SIZE = (128, 128)  # 结构尺寸边界保留给菜单显式声明。 / EN: keep single explicitly declare.
    RUN_PREFLIGHT_CHECKS = True  # 输入端消融专用：是否运行训练前轻量检查。 / EN: input-side ablation use: is runtrain before check.
    FULL_REFERENCE_MODEL_NAME = "MFPCHFNetV2_Full"  # 输入端消融专用：只读导入的 Full 参考模型名。 / EN: input-side ablation use: only read Full.


INPUT_ABLATION_CONFIG = TrainingConfig(
    name="input_ablation",
    active_entrypoint="Menu_InputAblation.py",
    manifest_filename="input_ablation_manifest.json",
    summary_prefix="input_ablation",
)


# ================= 输入端消融菜单 =================
# EN: ================= input-side ablation single =================.
# 逻辑 / Logic:
# EN: logic / Logic:
# 1. 顺序固定为 NIR -> Hyper -> Hyper+NIR -> Image -> Image+NIR -> Image+Hyper；
# EN: order order fixed as NIR -> Hyper -> Hyper+NIR -> Image -> Image+NIR -> Image+Hyper;
# 2. Full 不进入训练队列，只作为结果汇总时的只读参考来源；
# EN: Full not train column, only as result summary when only read source;
# 3. 不含图像的 NIR / Hyper / Hyper+NIR 使用 batch_size=256，提高无图像输入组合训练响应速度；
# EN: not image NIR / Hyper / Hyper+NIR use batch_size=256, high no image train should degree;
# 4. 图像相关组合继续使用 batch_size=10，避免 1024 图像分支训练显存压力过高；
# EN: image use batch_size=10, avoid 1024 image train high;
# 5. 图像相关组合必须显式声明 Full 结构标签；非图像组合不声明结构，输出记录写 none；
# EN: image mustexplicitly declare Full architecture label; non-image not result, write none;
# 6. 最近修改时间：2026-05-30；作者：ljy。
# EN: Last modified: 2026-05-30; Author: ljy.
INPUT_ABLATION_ALL_MODEL_SPECS = [
    ModelSpec(
        name="InputAblation_NIROnly",
        display_name="MFPC-HFNet-NIROnly",
        model_family="input_ablation",
        active_inputs=("nir",),
        priors_path=Config.PCA_PRIORS_PATH,
        batch_size=256,
        image_size=Config.FULL_IMAGE_SIZE,
    ),
    ModelSpec(
        name="InputAblation_HyperOnly",
        display_name="MFPC-HFNet-HyperOnly",
        model_family="input_ablation",
        active_inputs=("hyper",),
        priors_path=Config.PCA_PRIORS_PATH,
        batch_size=256,
        image_size=Config.FULL_IMAGE_SIZE,
    ),
    ModelSpec(
        name="InputAblation_HyperNIR",
        display_name="MFPC-HFNet-Hyper+NIR",
        model_family="input_ablation",
        active_inputs=("hyper", "nir"),
        priors_path=Config.PCA_PRIORS_PATH,
        batch_size=256,
        image_size=Config.FULL_IMAGE_SIZE,
    ),
    ModelSpec(
        name="InputAblation_ImageOnly",
        display_name="MFPC-HFNet-ImageOnly",
        model_family="input_ablation",
        active_inputs=("image",),
        structure="high1+high2+high3+low",
        priors_path=Config.PCA_PRIORS_PATH,
        batch_size=10,
        image_size=Config.FULL_IMAGE_SIZE,
    ),
    ModelSpec(
        name="InputAblation_ImageNIR",
        display_name="MFPC-HFNet-Image+NIR",
        model_family="input_ablation",
        active_inputs=("image", "nir"),
        structure="high1+high2+high3+low",
        priors_path=Config.PCA_PRIORS_PATH,
        batch_size=10,
        image_size=Config.FULL_IMAGE_SIZE,
    ),
    ModelSpec(
        name="InputAblation_ImageHyper",
        display_name="MFPC-HFNet-Image+Hyper",
        model_family="input_ablation",
        active_inputs=("image", "hyper"),
        structure="high1+high2+high3+low",
        priors_path=Config.PCA_PRIORS_PATH,
        batch_size=10,
        image_size=Config.FULL_IMAGE_SIZE,
    ),
]


INPUT_ABLATION_MENU = ExperimentMenu(
    config=INPUT_ABLATION_CONFIG,
    config_cls=Config,
    all_model_specs=INPUT_ABLATION_ALL_MODEL_SPECS,
    preset_groups={
        "single_only": ["InputAblation_NIROnly", "InputAblation_HyperOnly", "InputAblation_ImageOnly"],
        "pair_only": ["InputAblation_HyperNIR", "InputAblation_ImageNIR", "InputAblation_ImageHyper"],
    },
)


def get_input_ablation_menu() -> ExperimentMenu:
    """
    返回输入端消融训练菜单。
    English: returnInputtrainingmenu.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    INPUT_ABLATION_MENU.sync_to_engine()
    return INPUT_ABLATION_MENU


_MENU = get_input_ablation_menu()
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
    运行输入端消融训练菜单。
    English: Inputtrainingmenu.

    最近修改时间 / Last modified: 2026-05-29
    English: Last modified: 2026-05-29.
    作者 / Author: ljy
    English: Author: ljy.
    """

    return run_training_menu(_MENU)


__all__ = [
    "Config",
    "INPUT_ABLATION_ALL_MODEL_SPECS",
    "INPUT_ABLATION_CONFIG",
    "INPUT_ABLATION_MENU",
    "MODEL_SPECS",
    "ModelSpec",
    "get_input_ablation_menu",
    "main",
    "select_model_specs",
]


if __name__ == "__main__":
    main()
