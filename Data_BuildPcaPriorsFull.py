# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
# =============================================================================
# Data_BuildPcaPriorsFull.py - 通道归一化增强版
# EN: Data_BuildPcaPriorsFull.py - channel normalization.
#
# 代码大纲：
# EN: code outline:
# 1. 扫描原始样本目录，完成与训练工程一致的同级 blank 校正。
# EN: scan start sample directory, complete and training projectconsistent same blank.
# 2. 对校正后的 8 通道图像构建拉普拉斯金字塔，并按频层执行中心裁切。
# EN: for after 8 pass imagebuildLaplacian pyramid, and by frequency bandexecutecenter crop.
# 3. 先做“第一遍统计”：在各频层的向量空间内累计按通道 mean/std，
# EN: first do "": in each frequency band vector space inside accumulate by pass mean/std,.
#    构造固定图像通道归一化先验。
# EN: fixedimagechannel normalization first.
# 4. 再做“第二遍正式构建”：
# EN: then do "":
#    - 使用第一遍得到的固定归一化参数，对 high1/high2/high3/low 分别做
# EN: use section obtain fixed parameters, for high1/high2/high3/low do.
#      逐通道 z-score 标准化；
# EN: per-channel z-score;
#    - 高频层在归一化空间内执行结构向量筛选；
# EN: high-frequency levels in inside executestructural-vector selection;
#    - 低频层在归一化空间内直接保留全部向量；
# EN: low-frequency level in inside directlykeep full amount;
#    - 基于归一化后的向量拟合 PCA，得到固定 1x1 主成分映射权重。
# EN: after amount fit PCA, obtainfixed 1x1 principal componentsmapping weights.
# 5. 将“通道归一化参数 + PCA 参数”一并写入 pca_priors_full.pt，
# EN: " + PCA " and write pca_priors_full.pt,.
#    供训练/推理阶段在分离主成分卷积前自动复用。
# EN: train/ in principal components before automaticallyreuse.
#
# 关键设计考虑：
# EN: key design considerations:
# A. 当前版本不再把“通道尺度差异”混入 PCA 主成分估计中，从而避免高方差波段
# EN: current no longer ""mix into PCA principal componentsestimate in, therebyavoidhigh-variance bands.
#    在协方差空间内天然主导主成分方向。
# EN: in covariance space inside naturally dominatesprincipal components.
# B. 结构向量筛选也改为在归一化空间中完成，避免能量统计被大幅值通道带偏。
# EN: structural-vector selection also change as in in complete, avoid can amount large value pass.
# C. 显式保留 norm_center / norm_scale，而不是把它们提前折叠进 PCA weight/bias，
# EN: form keep norm_center / norm_scale, not is before fold into PCA weight/bias,.
#    这样训练代码、画图脚本与论文表述都更容易核对，也便于未来继续扩展。
# EN: training code, image and table all for, also to make it easier tofuture extend.
# D. 代码保留较详细注释，目的是后续回看时能快速恢复“为什么这样设计”。
# EN: code keep, is laterreview later when can resume"".
# E. 最近注释维护：2026-05-29；作者：ljy。补充离线先验构建关键函数的输入输出、
# EN: Latest comment maintenance: 2026-05-29; Author: ljy. offline priorsbuild function inputs and outputs,.
#    物理含义和复现注意事项，不改变筛选、PCA 或输出逻辑。
# EN: meaning and reproducibility notes, does not change, PCA or output logic.
# F. 最近修改时间：2026-06-17；作者：GG。新增训练期 Fold Train 子集先验构建函数，
# EN: Last modified: 2026-06-17; Author: GG. new train Fold Train first buildfunction,.
#    用于交叉验证中避免 Validation/Test 参与 PCA/归一化先验估计。
# EN: use cross-validation in avoid Validation/Test participate in PCA/ first estimate.
# =============================================================================

import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import re
import csv
import copy
import math
import json
import time
import random
import argparse
import traceback
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
import scipy.io as sio
# ================= 默认运行配置 =================
# EN: ================= default runtime configuration =================.
# 逻辑：
# EN: Logic:
# 1. 这些配置用于“直接点运行脚本”时的默认行为；
# EN: use "" when default line as;
# 2. 若不传命令行参数，则自动使用这里的设置；
# EN: if not command-line arguments, then automaticallyusehere;
# 3. analysis_target 可选：
# EN: analysis_target optional:
#    - "all"  : 不要求真值
# EN: "all": do not value.
#    - "soc"  : 只分析有 SOC 真值的样本
# EN: "soc": only SOC value sample.
#    - "tn"   : 只分析有 TN 真值的样本
# EN: "tn": only TN value sample.
#    - "both" : 只分析同时有 SOC 和 TN 真值的样本
# EN: "both": only same when SOC and TN value sample.
# 4. 所有输出统一保存到当前运行路径下的 ModelData 文件夹中；
# EN: save to currentrunpath below ModelData folder in;
# 5. 输出目录自动带当天日期，方便区分不同版本。
# EN: output directoryautomatically, not same.
# 6. 最近修改时间：2026-06-16；作者：ljy。公开版默认路径改为相对示例目录，真实路径通过命令行传入。
# EN: Last modified: 2026-06-16; Author: ljy.public releasedefaultpath change as for, path pass line pass in.
DEFAULT_DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ExampleData", "LegacyRawDatabase")
DEFAULT_DATA_ROOT = os.path.join(DEFAULT_DATASET_ROOT, "raw_samples")
DEFAULT_SOC_MAT_PATH = os.path.join(DEFAULT_DATASET_ROOT, "labels", "SOC.mat")
DEFAULT_TN_MAT_PATH = os.path.join(DEFAULT_DATASET_ROOT, "labels", "TN.mat")


DEFAULT_ANALYSIS_TARGET = "soc"   # all / soc / tn / both
DEFAULT_MAX_SAMPLES = None
DEFAULT_MAX_VISUAL_SAMPLES = 32

# ================= 影响“筛选数量 / 主成分数量”的核心参数区 =================
# EN: ================= " / " core parameter section =================.
# 使用建议：
# EN: Usage advice:
# 1. 这里不再只放参数名，而是按“真正影响数量的链路”分组；
# EN: hereno longeronly parameters name, is by "";
# 2. 后续若想调“高频结构向量保留数量”，优先只动 P1 组；
# EN: later if want "", preferonly P1;
# 3. 若 P1 调完仍不够，再动 P2 组；P3 组主要影响 PCA 主成分数；
# EN: if P1 still not, then P2; P3 need PCA principal components number;
# 4. DEFAULT_NORM_EPS 只负责数值稳定，通常不会直接改变筛选数量，不建议拿它调数量。
# EN: DEFAULT_NORM_EPS only valuestable, usually not will directly change selection count, not recommended count.
#
# 高频结构向量保留数量主链：
# EN: high kept structural-vector count link:
#   裁切区域大小 -> 总向量数 N -> 能量阈值初筛(chi2/snr) -> 背景分布估计 ->
# EN: crop region large small -> total vector count N -> initial energy-threshold screening(chi2/snr) -> background-distribution estimation ->.
#   马氏距离 + FDR -> 保底 keep 下限 -> 最终 kept_count
# EN: + FDR -> keep lower bound -> most kept_count.
#
# PCA 主成分数量主链：
# EN: PCA number of principal components link:
#   裁切区域 + 结构向量筛选 -> 进入 PCA 的向量集合 -> 协方差谱 -> eta / max_components -> selected_k
# EN: crop region + structural-vector selection -> PCA amount -> covariance spectrum -> eta / max_components -> selected_k.
# ============================================================
# 参数与“筛选数量”关系说明（建议直接放在参数设置区上方）
# EN: parameters and "" note(recommendeddirectly in parameter-setting section on)
# ============================================================
#
# 【一】本脚本里有两类“数量”，不要混淆
# EN: []this script "", do not.
#
# 1. 高频结构向量保留数量 kept_count
# EN: high kept structural-vector count kept_count.
#    含义：
# EN: Meaning:
#    指 high1 / high2 / high3 在完成能量检验、背景建模、Mahalanobis 距离检验、
# EN: high1 / high2 / high3 in completeenergy test, background modeling, Mahalanobis,.
#    FDR 多重校正以及保底约束后，最终被保留下来的结构向量数量。
# EN: FDR more minimum-keep constraint after, most keep below structural vectorscount.
#
# 2. PCA 主成分数量 selected_k
# EN: PCA number of principal components selected_k.
#    含义：
# EN: Meaning:
#    指某一频层最终进入 PCA 后，根据累计解释方差阈值 eta 和上限 max_components
# EN: frequency band most PCA after, data cumulative explained-variance threshold eta and upper bound max_components.
#    所确定的主成分保留数量。
# EN: confirm principal componentskeepcount.
#
# ------------------------------------------------------------
# 【二】高频结构向量数量 kept_count 的主控参数及正负关系
# EN: [] high structural vectorscount kept_count primary control parameters positive/negative relationship.
# ------------------------------------------------------------
#
# P1 级：第一优先级，真正直接控制 kept_count 的核心参数
# EN: P1: first priority, actuallydirectlycontrol kept_count parameters.
#
# 1. snr_db
#    作用：
# EN: Purpose:
#    控制噪声方差 sigma_n2 的估计。snr_db 越大，sigma_n2 越小，
# EN: controlnoise variance sigma_n2 estimate.snr_db large, sigma_n2 small,.
#    则 energy = sum(X^2) / sigma_n2 越大，越容易通过第一道能量阈值。
# EN: then energy = sum(X^2) / sigma_n2 large, pass section can amount threshold.
#    与 kept_count 的关系：
# EN: and kept_count:
#    snr_db 增大  -> kept_count 通常增大
# EN: snr_db increase -> kept_count usually large.
#    snr_db 减小  -> kept_count 通常减小
# EN: snr_db decrease -> kept_count usually small.
#    修改建议：
# EN: Tuning advice:
#    当感觉“筛掉太多”时，可优先增大 snr_db。
# EN: "" when, can prefer large snr_db.
#
# 2. chi2_quantile
#    作用：
# EN: Purpose:
#    控制第一道能量检验的卡方分位阈值 chi2_th。
# EN: control section energy test chi-square quantile threshold chi2_th.
#    分位数越高，阈值越高，越难通过能量门槛。
# EN: number high, threshold high, pass energy threshold.
#    与 kept_count 的关系：
# EN: and kept_count:
#    chi2_quantile 增大 -> kept_count 通常减小
# EN: chi2_quantile increase -> kept_count usually small.
#    chi2_quantile 减小 -> kept_count 通常增大
# EN: chi2_quantile decrease -> kept_count usually large.
#    修改建议：
# EN: Tuning advice:
#    当希望放宽第一道筛选时，可适度降低 chi2_quantile。
# EN: section when, can moderately decrease chi2_quantile.
#
# 3. fdr_q
#    作用：
# EN: Purpose:
#    控制 Mahalanobis 距离显著性检验后的 Benjamini-Hochberg FDR 校正宽松程度。
# EN: control Mahalanobis significance test after Benjamini-Hochberg FDR program degree.
#    fdr_q 越大，允许保留的显著向量越多。
# EN: fdr_q large, allowkeep amount more.
#    与 kept_count 的关系：
# EN: and kept_count:
#    fdr_q 增大 -> kept_count 通常增大
# EN: fdr_q increase -> kept_count usually large.
#    fdr_q 减小 -> kept_count 通常减小
# EN: fdr_q decrease -> kept_count usually small.
#    修改建议：
# EN: Tuning advice:
#    当能量检验后还有不少候选，但最终保留仍太少时，优先检查 fdr_q。
# EN: energy test after not fewer, most keep still too few when, prefercheck fdr_q.
#
# ------------------------------------------------------------
# 【三】高频结构向量数量 kept_count 的保底参数
# EN: [] high structural vectorscount kept_count minimum-keep parameters.
# ------------------------------------------------------------
#
# P1 级：当 FDR 结果过少时，这组参数会“强行抬高”最终保留数量
# EN: P1: FDR result result fewer when, parameters will "" most keepcount.
#
# 4. selection_min_keep_ratio
#    作用：
# EN: Purpose:
#    设定最终至少保留的比例下限 min_keep_ratio * 候选总数。
# EN: most fewer keep ratio lower bound min_keep_ratio * candidate count.
#    仅在 FDR 后数量低于该下限时生效。
# EN: only in FDR after count low this lower bound when.
#    与 kept_count 的关系：
# EN: and kept_count:
#    selection_min_keep_ratio 增大 -> kept_count 在触发保底时增大
# EN: selection_min_keep_ratio increase -> kept_count in when large.
#    selection_min_keep_ratio 减小 -> kept_count 在触发保底时减小
# EN: selection_min_keep_ratio decrease -> kept_count in when small.
#    注意：
# EN: Note:
#    如果当前未触发保底，则该参数对结果无影响。
# EN: result current not yet, then this parameters for result result no.
#
# 5. selection_min_keep_channel_multiplier
#    作用：
# EN: Purpose:
#    设定与通道数相关的最小保留数下限。
# EN: and pass number minimum kept-count lower bound.
#    最终 min_keep 会综合比例下限、通道倍数下限和绝对下限共同决定。
# EN: most min_keep will ratio lower bound, channel-multiplier lower bound and for lower bound same.
#    与 kept_count 的关系：
# EN: and kept_count:
#    selection_min_keep_channel_multiplier 增大 -> kept_count 在触发保底时增大
# EN: selection_min_keep_channel_multiplier increase -> kept_count in when large.
#    selection_min_keep_channel_multiplier 减小 -> kept_count 在触发保底时减小
# EN: selection_min_keep_channel_multiplier decrease -> kept_count in when small.
#    注意：
# EN: Note:
#    仅在触发保底时生效。
# EN: only in when.
#
# 6. selection_min_keep_abs
#    作用：
# EN: Purpose:
#    设定绝对最小保留数量下限。
# EN: for minimumkeepcountlower bound.
#    与 kept_count 的关系：
# EN: and kept_count:
#    selection_min_keep_abs 增大 -> kept_count 在触发保底时增大
# EN: selection_min_keep_abs increase -> kept_count in when large.
#    selection_min_keep_abs 减小 -> kept_count 在触发保底时减小
# EN: selection_min_keep_abs decrease -> kept_count in when small.
#    注意：
# EN: Note:
#    仅在触发保底时生效。
# EN: only in when.
#
# ------------------------------------------------------------
# 【四】背景建模参数：会影响 kept_count，但通常不是单调正负关系
# EN: []background modelingparameters: will kept_count, usually not is single positive/negative relationship.
# ------------------------------------------------------------
#
# P2 级：第二优先级，主要影响“背景统计是否稳定”，不建议当作一阶调数量旋钮
# EN: P2: second priority, need "", not recommended count.
#
# 7. background_min_count
# 8. background_min_channel_multiplier
#    作用：
# EN: Purpose:
#    决定“背景候选是否足够”的判定门槛。
# EN: "".
#    过高时更容易触发背景样本不足，从而进入 fallback。
# EN: high when sample not meet, thereby fallback.
#    与 kept_count 的关系：
# EN: and kept_count:
#    不稳定、非严格单调
# EN: not stable, non- single.
#    说明：
# EN: Notes:
#    它们主要改变背景均值和协方差估计是否可靠，而不是直接决定保留数量。
# EN: need change mean and estimate is no can, not is directly keepcount.
#
# 9. background_fallback_base
# 10. background_fallback_channel_multiplier
#     作用：
# EN: Purpose:
#     当背景候选不足时，fallback 会补充更多低能量向量来构造背景集。
# EN: not meet when, fallback will more low can amount amount.
#     与 kept_count 的关系：
# EN: and kept_count:
#     不稳定、非严格单调
# EN: not stable, non- single.
#     说明：
# EN: Notes:
#     更多背景向量通常会让背景协方差估计更稳，但不保证 kept_count 一定增多或减少。
# EN: more amount usually will let estimate, not ensure kept_count more or fewer.
#
# 11. max_background_for_cov
#     作用：
# EN: Purpose:
#     限制用于协方差估计的背景向量上限。
# EN: use estimate amount upper bound.
#     与 kept_count 的关系：
# EN: and kept_count:
#     不稳定、非严格单调
# EN: not stable, non- single.
#     说明：
# EN: Notes:
#     更多样本一般有助于背景统计稳定，但不适合拿来直接调保留数量。
# EN: more sample stable, not suitable for directly keepcount.
#
# 12. cov_shrinkage_alpha
# 13. cov_shrinkage_eps
#     作用：
# EN: Purpose:
#     协方差矩阵正则化参数，主要防止协方差病态、奇异或数值不稳定。
# EN: then parameters, need prevent, or value not stable.
#     与 kept_count 的关系：
# EN: and kept_count:
#     不稳定、非严格单调
# EN: not stable, non- single.
#     说明：
# EN: Notes:
#     这两个参数优先服务于“稳定性”，不是直接服务于“数量”。
# EN: parametersprefer task "", not is directly task "".
#
# ------------------------------------------------------------
# 【五】PCA 主成分数量 selected_k 的主控参数及正负关系
# EN: []PCA number of principal components selected_k primary control parameters positive/negative relationship.
# ------------------------------------------------------------
#
# P1 级：第一优先级，真正直接控制 PCA 保留维数的核心参数
# EN: P1: first priority, actuallydirectlycontrol PCA keep number parameters.
#
# 14. eta_high
#     作用：
# EN: Purpose:
#     控制 high1 / high2 / high3 的累计解释方差目标。
# EN: control high1 / high2 / high3 accumulate.
#     eta_high 越大，为达到更高累计解释方差，需要保留更多主成分。
# EN: eta_high large, as to high accumulate, need need keep more principal components.
#     与 selected_k 的关系：
# EN: and selected_k:
#     eta_high 增大 -> selected_k 增大或不变
# EN: eta_high increase -> selected_k large or not.
#     eta_high 减小 -> selected_k 减小或不变
# EN: eta_high decrease -> selected_k small or not.
#
# 15. eta_low
#     作用：
# EN: Purpose:
#     控制 low 频层的累计解释方差目标。
# EN: control low frequency band accumulate.
#     与 selected_k 的关系：
# EN: and selected_k:
#     eta_low 增大 -> low 频层 selected_k 增大或不变
# EN: eta_low increase -> low frequency band selected_k large or not.
#     eta_low 减小 -> low 频层 selected_k 减小或不变
# EN: eta_low decrease -> low frequency band selected_k small or not.
#
# 16. max_components
#     作用：
# EN: Purpose:
#     控制 PCA 最多允许保留的主成分数量上限。
# EN: control PCA most more allowkeep number of principal componentsupper bound.
#     与 selected_k 的关系：
# EN: and selected_k:
#     max_components 增大 -> selected_k 增大或不变
# EN: max_components increase -> selected_k large or not.
#     max_components 减小 -> selected_k 减小或不变
# EN: max_components decrease -> selected_k small or not.
#     说明：
# EN: Notes:
#     这是一个“硬上限”，即使 eta 还希望保留更多维，也不能超过该值。
# EN: is "", make eta keep more, also not can this value.
#
# ------------------------------------------------------------
# 【六】与“进入 PCA 的向量总数”相关，但不等于直接控制 kept_count 的参数
# EN: [] and " PCA ", not directlycontrol kept_count parameters.
# ------------------------------------------------------------
#
# P2 级：第二优先级，主要影响样本基数，而不是直接改阈值
# EN: P2: second priority, need sample number, not is directly change threshold.
#
# 17. crop_high1 / crop_high2 / crop_high3
#     作用：
# EN: Purpose:
#     控制各高频层参与分析的图像区域范围，从而影响总向量数 N。
# EN: control each high-frequency levelsparticipate in image, thereby total vector count N.
#     与 kept_count 的关系：
# EN: and kept_count:
#     crop 范围增大 -> 候选总向量数通常增大 -> kept_count 绝对数量通常增大
# EN: crop increase -> total vector countusuallyincrease -> kept_count for countusually large.
#     crop 范围减小 -> 候选总向量数通常减小 -> kept_count 绝对数量通常减小
# EN: crop decrease -> total vector countusuallydecrease -> kept_count for countusually small.
#     注意：
# EN: Note:
#     这里通常影响的是“绝对数量”，不一定让保留比例同步上升。
# EN: hereusually is "", not let keep same on.
#
# 18. crop_low
#     作用：
# EN: Purpose:
#     仅影响 low 频层进入 PCA 的向量总数。
# EN: only low frequency band PCA amount number.
#     与 kept_count 的关系：
# EN: and kept_count:
#     对高频 kept_count 无直接作用
# EN: for high kept_count no directlypurpose.
#     与 selected_k 的关系：
# EN: and selected_k:
#     仅通过改变 low 频层 PCA 输入样本集合，间接影响 low 频层 selected_k
# EN: only pass change low frequency band PCA sample, indirectly low frequency band selected_k.
#
# ------------------------------------------------------------
# 【七】最实用的调参优先级建议
# EN: [] most use prefer recommended.
# ------------------------------------------------------------
#
# 1. 如果目标是“增加高频结构向量保留数量 kept_count”
# EN: result is " kept_count".
#    优先修改顺序：
# EN: prefer change order order:
#    snr_db
#    -> chi2_quantile
#    -> fdr_q
#    -> selection_min_keep_ratio / selection_min_keep_channel_multiplier / selection_min_keep_abs
#
# 2. 如果目标是“减少高频结构向量保留数量 kept_count”
# EN: result is " kept_count".
#    优先修改顺序：
# EN: prefer change order order:
#    chi2_quantile
#    -> fdr_q
#    -> snr_db
#    -> 再检查是否被 min_keep 保底强行托住
# EN: -> then check is no min_keep line.
#
# 3. 如果目标是“增加 PCA 主成分数量 selected_k”
# EN: result is " PCA selected_k".
#    优先修改顺序：
# EN: prefer change order order:
#    eta_high 或 eta_low
# EN: eta_high or eta_low.
#    -> max_components
#
# 4. 如果目标是“减少 PCA 主成分数量 selected_k”
# EN: result is " PCA selected_k".
#    优先修改顺序：
# EN: prefer change order order:
#    eta_high 或 eta_low
# EN: eta_high or eta_low.
#    -> max_components
#
# 5. 如果问题是“结果波动大、不稳定、不同批次数量漂移明显”
# EN: result is ", , ".
#    优先检查：
# EN: prefercheck:
#    background_* 参数
# EN: background_* parameters.
#    -> max_background_for_cov
#    -> cov_shrinkage_alpha / cov_shrinkage_eps
#
# ------------------------------------------------------------
# 【八】一句话总结
# EN: [] result.
# ------------------------------------------------------------
#
# 1. kept_count 的第一控制链：
# EN: kept_count section control link:
#    snr_db, chi2_quantile, fdr_q, min_keep_*
#
# 2. selected_k 的第一控制链：
# EN: selected_k section control link:
#    eta_high / eta_low, max_components
#
# 3. background_* 和 covariance shrinkage 主要负责“稳定性”，
# EN: background_* and covariance shrinkage need "",.
#    不适合作为第一优先级的“数量调节旋钮”。
# EN: not suitable for as first priority "".
#
# ============================================================
# ================= P1：最优先修改，最直接影响“高频结构向量保留数量” =================
# EN: ================= P1: most prefer change, most directly "" =================.
# 逻辑：
# EN: Logic:
# 1. crop_* 先决定每个频层进入筛选的总向量数 N；
# EN: crop_* first each frequency band total vector count N;
# 2. snr_db 与 chi2_quantile 共同决定第一道能量阈值，直接影响 background_candidate 的构成；
# EN: snr_db and chi2_quantile same section can amount threshold, directly background_candidate;
# 3. fdr_q 决定 BH-FDR 的放宽/收紧程度，直接影响 structure_mask 的大小；
# EN: fdr_q BH-FDR / program degree, directly structure_mask large small;
# 4. min_keep_* 是最后的“强制保底”，会在 FDR 过严时直接接管最终 kept_count；
# EN: min_keep_* is most after "", will in FDR when directly most kept_count;
# 5. 因此若你的目标是“把保留结构向量数量调多/调少”，先按这组参数调整。
# EN: therefore if is "/", first by parameters.
DEFAULT_CROP_HIGH1 = 800
DEFAULT_CROP_HIGH2 = 400
DEFAULT_CROP_HIGH3 = 200
DEFAULT_CROP_LOW = None

DEFAULT_SNR_DB = 5.0
DEFAULT_CHI2_QUANTILE = 0.999
DEFAULT_FDR_Q = 0.01
DEFAULT_SELECTION_MIN_KEEP_RATIO = 0.01
DEFAULT_SELECTION_MIN_KEEP_CHANNEL_MULTIPLIER = 8
DEFAULT_SELECTION_MIN_KEEP_ABS = 32

# ================= P2：次优先修改，影响“高频结构向量筛选的稳定性与背景建模” =================
# EN: ================= P2: prefer change, "" =================.
# 逻辑：
# EN: Logic:
# 1. 这组参数通常不会像 P1 一样线性地改变 kept_count；
# EN: parametersusually not will image P1 property change kept_count;
# 2. 但它们会改变背景均值/协方差估计质量，从而间接改变马氏距离分布与最终 kept_count；
# EN: will change mean/ estimate amount, therebyindirectly change and most kept_count;
# 3. 当你发现不同批次 kept_count 波动大、或阈值已经改了但数量不稳定时，再考虑调这组。
# EN: not same kept_count large, or threshold already change count not stable when, then.
DEFAULT_BACKGROUND_MIN_COUNT = 64
DEFAULT_BACKGROUND_MIN_CHANNEL_MULTIPLIER = 4
DEFAULT_BACKGROUND_FALLBACK_BASE = 512
DEFAULT_BACKGROUND_FALLBACK_CHANNEL_MULTIPLIER = 32
DEFAULT_MAX_BACKGROUND_FOR_COV = 200000
DEFAULT_COV_SHRINKAGE_ALPHA = 0.10
DEFAULT_COV_SHRINKAGE_EPS = 1e-6

# ================= P3：最直接影响“PCA 保留主成分数量” =================
# EN: ================= P3: most directly "PCA " =================.
# 逻辑：
# EN: Logic:
# 1. eta_high / eta_low 是累计解释方差阈值，直接决定 selected_k 的下限需求；
# EN: eta_high / eta_low is cumulative explained-variance threshold, directly selected_k lower bound need;
# 2. max_components 是显式上限，会在 eta 已满足后继续硬截断；
# EN: max_components is form upper bound, will in eta already meet meet after;
# 3. 但它们作用的前提，是前面进入 PCA 的向量集合已经确定，因此它们排在 P1/P2 之后理解更准确。
# EN: purpose before, is before PCA amount already confirm, therefore in P1/P2 after confirm.
DEFAULT_ETA_HIGH = 0.99
DEFAULT_ETA_LOW = 0.995
DEFAULT_MAX_COMPONENTS = None

# ================= 其他稳定性参数 =================
# EN: ================= its stable property parameters =================.
DEFAULT_NORM_EPS = 1e-6
DEFAULT_COMPARE_TOPK = 3
DEFAULT_SEED = 42

BASE_RUN_DIR = os.getcwd()
DEFAULT_MODEL_DATA_DIR = os.path.join(BASE_RUN_DIR, "ModelData")

RUN_DATE_STR = time.strftime("%Y-%m-%d")
DEFAULT_OUTPUT_DIR = os.path.join(
    DEFAULT_MODEL_DATA_DIR,
    f"{RUN_DATE_STR}_PCA分析输出_{DEFAULT_ANALYSIS_TARGET.upper()}"
)

# ================= 全局说明 =================
# EN: ================= full note =================.
# 逻辑：
# EN: Logic:
# 1. 本脚本用于离线构造 MF-PCSparseConv 所需 PCA 主成分卷积先验；
# EN: this script use MF-PCSparseConv need PCA principal components first;
# 2. 与当前工程保持一致，采用“同级 blank 校正”；
# EN: and current program consistent, use " blank ";
# 3. 在各频层向量空间中新增固定图像通道归一化先验；
# EN: in each frequency bandvector space in new fixedimagechannel normalization first;
# 4. 不输出图片文件，只保存数值缓存，方便后续任意作图；
# EN: not image file, onlysavevaluecache, later task image;
# 5. 除输出模型构造所需 weight / bias 外，还额外保存：
# EN: need weight / bias, save:
#    - 样本级统计
# EN: sample.
#    - 频层级统计
# EN: frequency band.
#    - PCA 全谱信息
# EN: PCA full.
#    - 代表样本数值缓存
# EN: table sample count value cache.
#    - 跨频层子空间差异分析数据
# EN: frequency band data.
# 5. 推荐用途：
# EN: use:
#    - 模型读取 pca_priors_full.pt 直接初始化 PCA 1x1 卷积
# EN: read pca_priors_full.pt directly start PCA 1x1.
#    - 论文作图读取 csv / npz / json 直接生成不同图
# EN: image read csv / npz / json directlygenerate not same image.


# ================= 1. 基础 I/O 与工程一致性函数 =================
# EN: ================= 1. I/O and program consistent property function =================.
# 逻辑：
# EN: Logic:
# 1. 尽量与 dataset_loader.py 当前逻辑保持一致；
# EN: amount and dataset_loader.py currentlogic consistent;
# 2. 支持中文路径；
# EN: in path;
# 3. 样本唯一性判定基于“完整样本名”，而非 CoreID；
# EN: samplesingle property "", non- CoreID;
# 4. 图像分析阶段仅处理 8 通道图像，不在这里使用 Hyper/NIR 进行 PCA。
# EN: image only 8 pass image, not in hereuse Hyper/NIR line PCA.
WAVELENGTHS = ['0490', '0540', '0590', '0660', '0775', '0880', '0945', '1000']
WAVELENGTHS_NUM = np.array([490, 540, 590, 660, 775, 880, 945, 1000], dtype=np.float32)


def set_seed(seed: int):
    """
    固定离线先验构建中的随机过程。
    English: build.

    输入:
    English: Input:
        seed: 随机种子。
        English: seed: .
    作用:
    English: :
        同时设置 Python random、NumPy 和 PyTorch，保证代表样本抽样、背景子采样等步骤可复现。
        English: Python random, NumPy PyTorch, ensuresample, .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    """
    确保输出目录存在。
    English: Outputdirectory.

    输入:
    English: Input:
        path: 需要创建的目录路径。
        English: path: createdirectorypath.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str):
    """
    将 Python 字典保存为 UTF-8 JSON。
    English: Python dictionarysave UTF-8 JSON.

    输入:
    English: Input:
        obj: 待保存的结构化对象。
        English: obj: save.
        path: 输出 JSON 文件路径。
        English: path: Output JSON filepath.
    说明:
    English: :
        ensure_ascii=False 用于保留中文字段，便于后续直接阅读和论文复核。
        English: ensure_ascii=False field, .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_ground_truth_dicts(soc_mat_path=None, tn_mat_path=None):
    """
    逻辑：
    English: Logic:
    1. 按需读取 SOC.mat / TN.mat；
    English: TN.mat；.
    2. 返回两个字典，键均为 CoreID；
    English: 2. returndictionary, CoreID;
    3. 若某个路径未提供，则对应字典为空。
    English: 3. path, dictionary.
    """
    soc_dict = {}
    tn_dict = {}

    if soc_mat_path is not None:
        soc_data = sio.loadmat(soc_mat_path)
        soc_sample_names = [str(x[0][0]).strip() for x in soc_data['SampleName']]
        soc_values = soc_data['SOC_Value'].flatten().astype(np.float32)
        soc_dict = dict(zip(soc_sample_names, soc_values))

    if tn_mat_path is not None:
        tn_data = sio.loadmat(tn_mat_path)
        tn_sample_names = [str(x[0][0]).strip() for x in tn_data['SampleName']]
        tn_values = tn_data['TN_Value'].flatten().astype(np.float32)
        tn_dict = dict(zip(tn_sample_names, tn_values))

    return soc_dict, tn_dict

def extract_core_id(folder_name):
    """
    逻辑：
    English: Logic:
    1. 文件夹名按 '-' 切分；
    English: 1. file '-' ;
    2. 去掉首段前缀（如 H / H5）；
    English: H5）；.
    3. 去掉末段重复号（要求为4位数字，如 0001）；
    English: 3. (4, 0001);
    4. 中间部分重新拼接，作为 CoreID。
    English: 4. , CoreID.
    """
    if not folder_name or '-' not in folder_name:
        return None

    parts = [part.strip() for part in folder_name.strip().split('-')]
    if len(parts) < 3:
        return None

    if not re.fullmatch(r"\d{4}", parts[-1]):
        return None

    core_parts = parts[1:-1]
    if not core_parts or any(part == '' for part in core_parts):
        return None

    return '-'.join(core_parts)


def read_hyper_csv(filepath):
    """
    读取单个样本的 HyperVISNIR.csv。
    English: readsample HyperVISNIR.csv.

    输入:
    English: Input:
        filepath: HyperVISNIR.csv 路径。
        English: filepath: HyperVISNIR.csv path.
    输出:
    English: Output:
        681 维 float32 数组；读取失败或长度不足时返回 None。
        English: 681 float32 ; readreturn None.

    说明:
    English: :
        该函数与训练数据读取保持一致，只取文件末尾 681 个有效数值作为 HyperVISNIR 特征。
        English: trainingread, file 681 HyperVISNIR .
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
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
    except Exception:
        return None


def read_nir_csv(filepath):
    """
    读取单个样本的 NIR.CSV。
    English: readsample NIR.CSV.

    输入:
    English: Input:
        filepath: NIR.CSV 路径。
        English: filepath: NIR.CSV path.
    输出:
    English: Output:
        float32 数组；读取失败时返回 None。
        English: float32 ; readreturn None.

    说明:
    English: :
        优先用 pandas 读取表格，失败时退回纯文本逗号/空白切分，兼容现场采集文件格式差异。
        English: pandas read, /, compatiblefile.
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
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
        except Exception:
            return None


def read_images(folder_path):
    """
    读取一个样本或 blank 文件夹中的 8 通道固定波长图像。
    English: readsample blank file 8 image.

    输入:
    English: Input:
        folder_path: 包含 Image-xxxxnm.tif 的样本目录。
        English: folder_path: Image-xxxxnm.tif sampledirectory.
    输出:
    English: Output:
        [H, W, 8] float32 图像堆栈。
        English: [H, W, 8] float32 image.

    关键约束:
    English: :
        8 个波段必须全部存在且尺寸一致；否则抛出异常，避免 PCA 先验混入不完整样本。
        English: 8 ; , avoid PCA sample.
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
    """
    images = []

    for wl in WAVELENGTHS:
        fname = f"Image-{wl}nm.tif"
        fpath = os.path.join(folder_path, fname)

        if not os.path.exists(fpath):
            if os.path.exists(folder_path):
                files = os.listdir(folder_path)
                for f in files:
                    if f.lower() == fname.lower():
                        fpath = os.path.join(folder_path, f)
                        break

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing image: {fname} in {folder_path}")

        img_array = np.fromfile(fpath, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"OpenCV 解码返回空值: {fpath}")

        if img.ndim != 2:
            raise ValueError(f"图像维度异常: {fpath}, shape={img.shape}")

        if img.size == 0:
            raise ValueError(f"图像为空: {fpath}")

        images.append(img.astype(np.float32))

    # 检查 8 张固定波段图像尺寸一致
    # EN: check 8 fixedbandsimage consistent.
    base_shape = images[0].shape
    for i, img in enumerate(images):
        if img.shape != base_shape:
            raise ValueError(
                f"同一样本不同波段尺寸不一致: folder={folder_path}, "
                f"band={WAVELENGTHS[i]}, shape={img.shape}, expected={base_shape}"
            )

    img_stack = np.stack(images, axis=-1)
    return img_stack.astype(np.float32)


def check_sample_files(folder_path):
    """
    检查样本目录是否具备构建先验所需的最小文件集合。
    English: checksampledirectorybuildfile.

    输入:
    English: Input:
        folder_path: 样本目录或 blank 目录。
        English: folder_path: sampledirectory blank directory.
    输出:
    English: Output:
        True 表示 HyperVISNIR.csv、NIR.CSV 和 8 张固定波长图像均存在。
        English: True HyperVISNIR.csv, NIR.CSV 8 image.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    hyper_path = os.path.join(folder_path, "HyperVISNIR.csv")
    nir_path = os.path.join(folder_path, "NIR.CSV")

    if not os.path.isfile(hyper_path):
        return False
    if not os.path.isfile(nir_path):
        return False

    try:
        files_lower = {f.lower() for f in os.listdir(folder_path)}
    except Exception:
        return False

    for wl in WAVELENGTHS:
        fname = f"Image-{wl}nm.tif".lower()
        if fname not in files_lower:
            return False

    return True


def center_crop_feature_map(feat_chw, crop_size=None):
    """
    逻辑：
    English: Logic:
    1. 输入为金字塔某一频层的特征图 [C, H, W]；
    English: 1. Input [C, H, W];
    2. 若 crop_size 为 None，则保持旧版行为，直接返回整张特征图；
    English: 2. crop_size None, , return;
    3. 若指定 crop_size，则仅保留中心 crop_size×crop_size 区域；
    English: 3. crop_size, crop_size×crop_size ;
    4. 这里专门用于“先金字塔，再裁切，再做向量筛选”的流程；
    English: 4. “, , ”;
    5. 若目标裁切尺寸大于当前特征图尺寸，则报错，避免静默使用错误 ROI。
    English: 5. current, , avoid ROI.
    """
    feat = feat_chw.astype(np.float32)

    if crop_size is None:
        return feat

    crop_size = int(crop_size)
    if crop_size <= 0:
        raise ValueError(f"crop_size 必须为正整数或 None，当前为: {crop_size}")

    c, h, w = feat.shape
    if crop_size > h or crop_size > w:
        raise ValueError(
            f"中心裁切尺寸超过当前频层尺寸，crop_size={crop_size}, feature_shape=({c}, {h}, {w})"
        )

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    bottom = top + crop_size
    right = left + crop_size

    return feat[:, top:bottom, left:right].astype(np.float32)


# ================= 2. 扫描数据集与 blank 校正 =================
# EN: ================= 2. scandata and blank =================.
# 逻辑：
# EN: Logic:
# 1. 扫描所有包含 blank 的批次目录；
# EN: scan blank;
# 2. 只保留关键文件齐全、样本名全局不重复的记录；
# EN: onlykeep file full, sample name full not;
# 3. blank 校正与当前工程一致：sample / blank；
# EN: blank and current program consistent: sample / blank;
# 4. 这里不依赖真值文件，因为当前任务是图像统计先验构造。
# EN: here not ground-truth file, becausecurrent task task is image first.
class RawImageSampleScanner:
    """
    原始图像样本扫描器。
    English: imagesample.

    职责:
    English: :
        1. 按包含 blank 的批次目录扫描样本；
        English: 1. blank directorysample;
        2. 用完整样本名排除重复记录；
        English: 2. sample;
        3. 按 analysis_target 过滤 SOC/TN 真值可用性；
        English: 3. analysis_target SOC/TN ground truth;
        4. 迭代输出经过同级 blank 校正后的 8 通道图像。
        English: 4. Output blank 8 image.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    def __init__(self, data_root: str):
        """
        保存原始样本库根目录。
        English: savesampledirectory.

        输入:
        English: Input:
            data_root: 原始样本库根目录。
            English: data_root: sampledirectory.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        self.data_root = data_root

    def scan(self, analysis_target='all', soc_dict=None, tn_dict=None):
        """
        扫描样本并生成有效记录。
        English: sample.

        输入:
        English: Input:
            analysis_target: all/soc/tn/both，用于按真值可用性筛选样本。
            English: analysis_target: all/soc/tn/both, ground truthsample.
            soc_dict / tn_dict: CoreID 到 SOC/TN 真值的映射。
            English: tn_dict: CoreID 到 SOC/TN 真值的映射.
        输出:
        English: Output:
            valid_records: 可用于先验构建的样本记录。
            English: valid_records: buildsample.
            report: 扫描统计摘要。
            English: report: .
            duplicate_items: 因完整样本名重复而排除的记录。
            English: duplicate_items: sample.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        all_sample_name_records = {}
        scanned_groups = 0

        soc_dict = soc_dict or {}
        tn_dict = tn_dict or {}
        analysis_target = str(analysis_target).lower().strip()

        if analysis_target not in ('soc', 'tn', 'both', 'all'):
            raise ValueError(
                f"analysis_target 仅支持 'soc' / 'tn' / 'both' / 'all'，当前为: {analysis_target}"
            )

        for root, dirs, _ in os.walk(self.data_root):
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

                sample_folder_path = os.path.join(root, d)
                if sample_name not in all_sample_name_records:
                    all_sample_name_records[sample_name] = []

                all_sample_name_records[sample_name].append({
                    "root": root,
                    "blank_path": blank_path,
                    "folder_name": sample_name,
                    "folder_path": sample_folder_path,
                    "core_id": core_id,
                    "sample_name": sample_name
                })

        duplicate_sample_names = {
            sample_name for sample_name, records in all_sample_name_records.items()
            if len(records) > 1
        }

        valid_records = []
        duplicate_items = []

        for sample_name, records in all_sample_name_records.items():
            if sample_name in duplicate_sample_names:
                duplicate_items.extend(records)
                continue

            for record in records:
                if not check_sample_files(record["folder_path"]):
                    continue
                if not check_sample_files(record["blank_path"]):
                    continue

                core_id = record["core_id"]
                has_soc = core_id in soc_dict
                has_tn = core_id in tn_dict

                if analysis_target == 'soc' and not has_soc:
                    continue
                if analysis_target == 'tn' and not has_tn:
                    continue
                if analysis_target == 'both' and not (has_soc and has_tn):
                    continue

                record["has_soc"] = has_soc
                record["has_tn"] = has_tn

                if has_soc:
                    record["soc_value"] = float(soc_dict[core_id])
                if has_tn:
                    record["tn_value"] = float(tn_dict[core_id])

                valid_records.append(record)

        report = {
            "scanned_groups": int(scanned_groups),
            "unique_sample_names": int(len(all_sample_name_records)),
            "duplicate_sample_names": int(len(duplicate_sample_names)),
            "duplicate_items": int(len(duplicate_items)),
            "valid_records": int(len(valid_records)),
            "analysis_target": analysis_target,
            "valid_has_soc_count": int(sum(1 for x in valid_records if x.get("has_soc", False))),
            "valid_has_tn_count": int(sum(1 for x in valid_records if x.get("has_tn", False))),
            "valid_has_both_count": int(
                sum(1 for x in valid_records if x.get("has_soc", False) and x.get("has_tn", False))
            )
        }

        return valid_records, report, duplicate_items

    def iter_calibrated_images(self, records, max_samples=None):
        """
        逐个读取样本并执行同级 blank 校正。
        English: readsample blank .

        输入:
        English: Input:
            records: scan() 返回的有效样本记录。
            English: records: scan() returnsample.
            max_samples: 最多输出样本数；None 表示全部。
            English: max_samples: Outputsample; None .
        输出:
        English: Output:
            生成器，每次 yield 一个包含 image、sample_name、core_id 和真值可用性信息的字典。
            English: , yield image, sample_name, core_id ground truthdictionary.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        count = 0

        for record in records:
            try:
                blank_path = record["blank_path"]
                sample_path = record["folder_path"]

                b_imgs = read_images(blank_path)
                s_imgs = read_images(sample_path)

                if s_imgs.shape != b_imgs.shape:
                    continue

                # 与现有工程一致：同级 blank 校正
                # EN: and program consistent: same blank.
                img_cal = s_imgs / (b_imgs + 1e-8)  # [H, W, 8]
                img_chw = np.transpose(img_cal, (2, 0, 1)).astype(np.float32)  # [8, H, W]

                yield {
                    "image": img_chw,
                    "sample_name": record["sample_name"],
                    "core_id": record["core_id"],
                    "folder_path": record["folder_path"],
                    "blank_path": blank_path,
                    "root": record["root"],
                    "has_soc": record.get("has_soc", False),
                    "has_tn": record.get("has_tn", False),
                    "soc_value": record.get("soc_value", None),
                    "tn_value": record.get("tn_value", None)
                }

                count += 1
                if max_samples is not None and count >= max_samples:
                    break

            except Exception:
                continue


# ================= 3. 拉普拉斯金字塔分解 =================
# EN: ================= 3. Laplacian pyramid =================.
# 逻辑：
# EN: Logic:
# 1. 先高斯平滑再 2 倍下采样；
# EN: first high then 2 below;
# 2. 高频层 = 当前层 - 上采样对齐后的下一层；
# EN: high-frequency levels = current - on alignment after below;
# 3. 默认构建 3 个高频层和 1 个低频层；
# EN: defaultbuild 3 high-frequency levels and 1 low-frequency level;
# 4. 返回字典，方便后续统一处理。
# EN: return, later.
def build_gaussian_kernel():
    """
    构建离线拉普拉斯金字塔使用的 5x5 Gaussian kernel。
    English: build 5x5 Gaussian kernel.

    输出:
    English: Output:
        [5, 5] float32 卷积核，权重和为 1。
        English: [5, 5] float32 , 1.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    kernel = np.array(
        [[1., 4., 6., 4., 1.],
         [4., 16., 24., 16., 4.],
         [6., 24., 36., 24., 6.],
         [4., 16., 24., 16., 4.],
         [1., 4., 6., 4., 1.]],
        dtype=np.float32
    )
    kernel /= kernel.sum()
    return kernel


def smooth_image(img_chw, kernel):
    """
    对 [C, H, W] 图像逐通道执行高斯平滑。
    English: [C, H, W] image.

    输入:
    English: Input:
        img_chw: 8 通道图像或频层特征图。
        English: img_chw: 8 image.
        kernel: build_gaussian_kernel() 返回的二维核。
        English: kernel: build_gaussian_kernel() return.
    输出:
    English: Output:
        与输入同尺寸的 float32 平滑结果。
        English: Input float32 result.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    c, _, _ = img_chw.shape
    out = np.empty_like(img_chw, dtype=np.float32)
    for i in range(c):
        out[i] = cv2.filter2D(img_chw[i], -1, kernel, borderType=cv2.BORDER_REFLECT)
    return out


def pyramid_down(img_chw, kernel):
    """
    执行一次 Gaussian smoothing + 2 倍下采样。
    English: Gaussian smoothing + 2 .

    输入:
    English: Input:
        img_chw: [C, H, W] 图像。
        kernel: Gaussian kernel。
    输出:
    English: Output:
        [C, H/2, W/2] 下采样结果。
        English: [C, H/2, W/2] result.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    smoothed = smooth_image(img_chw, kernel)
    down = smoothed[:, ::2, ::2]
    return down


def pyramid_up(img_chw, target_hw):
    """
    将低分辨率图像上采样到指定空间尺寸。
    English: image.

    输入:
    English: Input:
        img_chw: [C, H, W] 图像。
        English: img_chw: [C, H, W] image.
        target_hw: 目标 (H, W)。
        English: target_hw: (H, W).
    输出:
    English: Output:
        [C, target_H, target_W] 双线性插值结果。
        English: [C, target_H, target_W] result.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    c = img_chw.shape[0]
    th, tw = target_hw
    up = np.empty((c, th, tw), dtype=np.float32)
    for i in range(c):
        up[i] = cv2.resize(img_chw[i], (tw, th), interpolation=cv2.INTER_LINEAR)
    return up


def build_laplacian_pyramid(img_chw, num_levels=3):
    """
    构建 3 层高频 + 1 层低频的拉普拉斯金字塔。
    English: build 3 + 1 .

    输入:
    English: Input:
        img_chw: blank 校正后的 [8, H, W] 图像。
        English: img_chw: blank [8, H, W] image.
        num_levels: 高频层数，当前正式口径为 3。
        English: num_levels: , current 3.
    输出:
    English: Output:
        包含 high1/high2/high3/low 以及中间 Gaussian 层的字典。
        English: high1/high2/high3/low Gaussian dictionary.

    物理意义:
    English: Physical meaning:
        high 层保留不同尺度的纹理和边缘扰动，low 层保留大尺度背景趋势。
        English: high , low .
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
    """
    current = img_chw.astype(np.float32)
    kernel = build_gaussian_kernel()

    highs = []
    gaussians = [current]

    for _ in range(num_levels):
        down = pyramid_down(current, kernel)
        up = pyramid_up(down, current.shape[1:])
        high = current - up
        highs.append(high.astype(np.float32))
        current = down.astype(np.float32)
        gaussians.append(current)

    low = current.astype(np.float32)

    return {
        "high1": highs[0],
        "high2": highs[1],
        "high3": highs[2],
        "low": low,
        "gaussian0": gaussians[0],
        "gaussian1": gaussians[1],
        "gaussian2": gaussians[2],
        "gaussian3": gaussians[3]
    }


# ================= 4. 高频结构向量筛选 =================
# EN: ================= 4. high structural-vector selection =================.
# 逻辑：
# EN: Logic:
# 1. 高频层中大量响应接近 0，需要先剔除背景与冗余；
# EN: high-frequency levels in large amount should 0, need need first and;
# 2. 先用卡方能量阈值构造背景候选；
# EN: first use can amount threshold;
# 3. 再用鲁棒均值 + 收缩协方差估计背景分布；
# EN: then use mean + estimate;
# 4. 对全部向量计算马氏距离平方并做 BH-FDR；
# EN: for full amount and do BH-FDR;
# 5. 返回保留向量、结构掩膜及完整统计量。
# EN: returnkeep amount, result complete amount.
def robust_center_median(X):
    """
    用逐通道中位数估计背景分布中心。
    English: This docstring documents the corresponding function behavior and engineering constraints.

    输入:
    English: Input:
        X: [N, C] 向量矩阵。
        English: X: [N, C] .
    输出:
    English: Output:
        [C] 鲁棒中心向量。
        English: [C] .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    return np.median(X, axis=0)


def shrinkage_covariance(X, eps=1e-6, alpha=0.1):
    """
    逻辑：
    English: Logic:
    1. alpha 越大，协方差越向各向同性单位阵收缩，背景模型越保守；
    English: 1. alpha , , model;
    2. eps 仅用于数值稳定，通常不应作为“调筛选数量”的主旋钮；
    English: 2. eps , “”;
    3. 这两个参数不会像 P1 参数那样直接决定 kept_count，
    English: 3. parameter P1 parameter kept_count,.
       但会通过改变背景协方差估计，间接影响马氏距离与 FDR 结果。
       English: , FDR result.
    """
    if X.shape[0] <= 1:
        c = X.shape[1]
        return np.eye(c, dtype=np.float32)

    cov = np.cov(X, rowvar=False).astype(np.float64)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)

    c = cov.shape[0]
    trace_avg = np.trace(cov) / max(c, 1)
    shrunk = (1.0 - alpha) * cov + alpha * np.eye(c, dtype=np.float64) * trace_avg
    shrunk += np.eye(c, dtype=np.float64) * eps
    return shrunk.astype(np.float32)


def mahalanobis_squared(X, mu, cov):
    """
    计算每个向量相对于背景分布的马氏距离平方。
    English: calculate.

    输入:
    English: Input:
        X: [N, C] 待检验向量。
        English: X: [N, C] .
        mu: [C] 背景中心。
        English: mu: [C] .
        cov: [C, C] 背景协方差。
        English: cov: [C, C] .
    输出:
    English: Output:
        [N] 马氏距离平方。
        English: [N] .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    diff = X - mu[None, :]
    inv_cov = np.linalg.pinv(cov).astype(np.float64)
    d2 = np.einsum('nc,cd,nd->n', diff, inv_cov, diff)
    return d2.astype(np.float64)


def benjamini_hochberg_threshold(pvals, q=0.05):
    """
    执行 Benjamini-Hochberg FDR 多重校正。
    English: Benjamini-Hochberg FDR .

    输入:
    English: Input:
        pvals: 每个向量的 p-value。
        English: pvals: p-value.
        q: 允许的 FDR 水平。
        English: q: FDR .
    输出:
    English: Output:
        p_cut: 通过校正的最大 p-value；无通过项时为 None。
        English: p_cut: p-value; None.
        mask: 与 pvals 同长度的 bool 保留掩膜。
        English: mask: pvals bool .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    m = len(pvals)
    if m == 0:
        return None, np.zeros((0,), dtype=bool)

    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresh = q * (np.arange(1, m + 1) / m)

    passed = sorted_p <= thresh
    if not np.any(passed):
        return None, np.zeros((m,), dtype=bool)

    k = np.max(np.where(passed)[0])
    p_cut = float(sorted_p[k])
    mask = pvals <= p_cut
    return p_cut, mask


def quantile_stats(x, prefix):
    """
    计算一组数值的常用分位数并展开为字典字段。
    English: calculatedictionaryfield.

    输入:
    English: Input:
        x: 一维数值数组。
        English: x: .
        prefix: 输出字段前缀。
        English: prefix: Outputfield.
    输出:
    English: Output:
        包含 q01/q05/.../q99 的字典。
        English: q01/q05/.../q99 dictionary.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    q = np.quantile(x, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        f"{prefix}_q01": float(q[0]),
        f"{prefix}_q05": float(q[1]),
        f"{prefix}_q10": float(q[2]),
        f"{prefix}_q25": float(q[3]),
        f"{prefix}_q50": float(q[4]),
        f"{prefix}_q75": float(q[5]),
        f"{prefix}_q90": float(q[6]),
        f"{prefix}_q95": float(q[7]),
        f"{prefix}_q99": float(q[8]),
    }


def summarize_channel_stats(X, prefix):
    """
    X: [N, C]
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    out = {}
    for i in range(X.shape[1]):
        out[f"{prefix}_ch{i:02d}_mean"] = float(mean[i])
        out[f"{prefix}_ch{i:02d}_std"] = float(std[i])
    return out


def select_structural_vectors(
    feat_chw,
    snr_db=20.0,
    chi2_quantile=0.95,
    fdr_q=0.05,
    min_keep_ratio=0.01,
    min_keep_channel_multiplier=8,
    min_keep_abs=32,
    background_min_count=64,
    background_min_channel_multiplier=4,
    background_fallback_base=512,
    background_fallback_channel_multiplier=32,
    max_background_for_cov=200000,
    cov_shrinkage_alpha=0.1,
    cov_shrinkage_eps=1e-6
):
    """
    输入：
        feat_chw: [C, H, W]
    输出：
        kept_vectors: [N_keep, C]
        info: 完整统计字典
        structure_mask: [H, W] bool
        aux: 额外数值缓存
        English: aux: cache.

    数量链路说明：
    English: :
    1. feat_chw 的空间尺寸先决定总向量数 N；
    English: 1. feat_chw N;
    2. snr_db + chi2_quantile 决定能量初筛阈值；
    English: 2. snr_db + chi2_quantile ;
    3. 背景相关参数决定用哪些低能量向量去估计背景分布；
    English: 3. parameter;
    4. fdr_q 决定马氏距离检验后的正式保留数量；
    English: 4. fdr_q ;
    5. 若正式保留数量过少，则 min_keep_* 会作为最终保底下限直接改写 kept_count。
    English: 5. , min_keep_* kept_count.
    """
    c, h, w = feat_chw.shape
    X = feat_chw.reshape(c, -1).T.astype(np.float32)  # [N, C]
    n = X.shape[0]

    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma_n2 = 1.0 / snr_lin

    # 能量统计
    # EN: can amount.
    energy = np.sum((X ** 2), axis=1) / (sigma_n2 + 1e-12)
    chi2_th = float(chi2.ppf(chi2_quantile, df=c))
    structure_candidate = energy > chi2_th
    background_candidate = ~structure_candidate

    bg_count = int(background_candidate.sum())
    background_min_required = max(int(background_min_count), int(c * background_min_channel_multiplier))
    background_fallback_take = min(
        max(int(background_fallback_base), int(c * background_fallback_channel_multiplier)),
        n
    )
    used_background_fallback = False

    if bg_count < background_min_required:
        order = np.argsort(energy)
        bg_idx = order[:background_fallback_take]
        background_candidate = np.zeros(n, dtype=bool)
        background_candidate[bg_idx] = True
        used_background_fallback = True

    X_bg = X[background_candidate]

    background_subsampled_for_cov = False
    if X_bg.shape[0] > max_background_for_cov:
        idx = np.random.choice(X_bg.shape[0], max_background_for_cov, replace=False)
        X_bg = X_bg[idx]
        background_subsampled_for_cov = True

    mu_hat = robust_center_median(X_bg)
    cov_hat = shrinkage_covariance(X_bg, eps=cov_shrinkage_eps, alpha=cov_shrinkage_alpha)

    d2 = mahalanobis_squared(X, mu_hat, cov_hat)
    pvals = 1.0 - chi2.cdf(d2, df=c)
    fdr_threshold_p, structure_mask = benjamini_hochberg_threshold(pvals, q=fdr_q)

    keep_count_before_floor = int(structure_mask.sum())
    min_keep = max(
        int(n * min_keep_ratio),
        int(c * min_keep_channel_multiplier),
        int(min_keep_abs)
    )
    used_min_keep_floor = keep_count_before_floor < min_keep

    if used_min_keep_floor:
        order = np.argsort(d2)[::-1]
        chosen = order[:min(min_keep, n)]
        structure_mask = np.zeros(n, dtype=bool)
        structure_mask[chosen] = True

    kept_vectors = X[structure_mask].astype(np.float32)

    info = {
        "total_vectors": int(n),
        "channel_dim": int(c),
        "snr_db": float(snr_db),
        "snr_linear": float(snr_lin),
        "sigma_n2": float(sigma_n2),
        "chi2_quantile": float(chi2_quantile),
        "chi2_threshold": float(chi2_th),
        "background_candidate_count_before_fallback": int(bg_count),
        "background_min_required": int(background_min_required),
        "background_fallback_take": int(background_fallback_take),
        "used_background_fallback": int(used_background_fallback),
        "background_candidate_count": int(background_candidate.sum()),
        "background_used_for_cov": int(X_bg.shape[0]),
        "background_subsampled_for_cov": int(background_subsampled_for_cov),
        "max_background_for_cov": int(max_background_for_cov),
        "cov_shrinkage_alpha": float(cov_shrinkage_alpha),
        "cov_shrinkage_eps": float(cov_shrinkage_eps),
        "fdr_q": float(fdr_q),
        "fdr_threshold_p": None if fdr_threshold_p is None else float(fdr_threshold_p),
        "keep_count_before_floor": int(keep_count_before_floor),
        "min_keep_ratio": float(min_keep_ratio),
        "min_keep_channel_multiplier": int(min_keep_channel_multiplier),
        "min_keep_abs": int(min_keep_abs),
        "min_keep_final": int(min_keep),
        "used_min_keep_floor": int(used_min_keep_floor),
        "kept_count": int(structure_mask.sum()),
        "kept_ratio": float(structure_mask.mean()),
        "energy_mean": float(energy.mean()),
        "energy_std": float(energy.std()),
        "energy_min": float(energy.min()),
        "energy_max": float(energy.max()),
        "d2_mean": float(d2.mean()),
        "d2_std": float(d2.std()),
        "d2_min": float(d2.min()),
        "d2_max": float(d2.max())
    }

    info.update(quantile_stats(energy, "energy"))
    info.update(quantile_stats(d2, "d2"))
    info.update(summarize_channel_stats(X, "before"))
    info.update(summarize_channel_stats(kept_vectors, "after"))

    aux = {
        "energy_map": energy.reshape(h, w).astype(np.float32),
        "d2_map": d2.reshape(h, w).astype(np.float32),
        "pval_map": pvals.reshape(h, w).astype(np.float32),
        "structure_mask": structure_mask.reshape(h, w),
        "robust_mean": mu_hat.astype(np.float32),
        "covariance": cov_hat.astype(np.float32)
    }

    return kept_vectors, info, structure_mask.reshape(h, w), aux


# ================= 5. PCA 拟合与主成分卷积参数构造 =================
# EN: ================= 5. PCA fit and principal components parameters =================.
# 逻辑：
# EN: Logic:
# 1. 对各频层收集到的向量做 PCA；
# EN: for each frequency band to amount do PCA;
# 2. 保存完整谱信息，避免后续重跑；
# EN: savecomplete, avoidlater;
# 3. 同时输出构造 1×1 主成分卷积所需 weight / bias；
# EN: same when 1x1 principal components need weight / bias;
# 4. 后续模型可直接读取并冻结使用；
# EN: later can directlyread and result use;
# 5. 为避免全局向量数极大时发生内存爆炸，这里新增“流式充分统计量”版本，
# EN: as avoid full amount number large when inside, here new "",.
#    不再要求先把所有样本向量整体拼成超大矩阵。
# EN: no longer need first sample amount large.


class RunningChannelNormStats:
    """
    逻辑：
    English: Logic:
    1. 用于第一遍扫描时累计按通道的 mean / std 所需充分统计量；
    English: std 所需充分统计量；.
    2. 这里统计的是“金字塔分频 + 中心裁切后”的向量空间，而不是原始整图空间；
    English: 2. “ + ”, ;
    3. 这样得到的归一化参数会与后续结构向量筛选、PCA 拟合以及模型前向保持一致；
    English: 3. parameter, PCA model;
    4. 当前采用 per-band、per-channel 的固定 z-score 归一化。
    English: 4. current per-band, per-channel z-score .
    """
    def __init__(self, channel_dim: int):
        """
        初始化通道统计量容器。
        English: This docstring documents the corresponding function behavior and engineering constraints.

        输入:
        English: Input:
            channel_dim: 频层向量的通道数，本工程固定为 8。
            English: channel_dim: , 8.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        self.channel_dim = int(channel_dim)
        self.count = 0
        self.sum_vec = np.zeros((self.channel_dim,), dtype=np.float64)
        self.sum_sq_vec = np.zeros((self.channel_dim,), dtype=np.float64)

    def update(self, X: np.ndarray):
        """
        累计一个批次向量的通道求和与平方和。
        English: This docstring documents the corresponding function behavior and engineering constraints.

        输入:
        English: Input:
            X: [N, C] 向量矩阵；空值或空矩阵会被忽略。
            English: X: [N, C] ; empty value.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if X is None:
            return
        X = np.asarray(X)
        if X.size == 0:
            return
        if X.ndim != 2:
            raise ValueError(f"RunningChannelNormStats.update 要求二维输入 [N, C]，当前维度为 {X.ndim}")
        if X.shape[1] != self.channel_dim:
            raise ValueError(
                f"通道数不一致，期望 {self.channel_dim}，当前为 {X.shape[1]}"
            )

        X32 = X.astype(np.float32, copy=False)
        self.count += int(X32.shape[0])
        self.sum_vec += X32.sum(axis=0, dtype=np.float64)
        self.sum_sq_vec += np.square(X32, dtype=np.float64).sum(axis=0, dtype=np.float64)

    def get_mean(self) -> np.ndarray:
        """
        输出累计样本的逐通道均值。
        English: Outputsample.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if self.count <= 0:
            raise ValueError("当前没有任何样本，无法计算通道均值")
        return (self.sum_vec / float(self.count)).astype(np.float64)

    def get_std(self, eps: float = 1e-6) -> np.ndarray:
        """
        输出累计样本的逐通道标准差。
        English: Outputsample.

        输入:
        English: Input:
            eps: 数值稳定项和最小标准差下限。
            English: eps: .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if self.count <= 0:
            raise ValueError("当前没有任何样本，无法计算通道标准差")
        mean = self.get_mean()
        second_moment = self.sum_sq_vec / float(self.count)
        var = np.maximum(second_moment - np.square(mean), 0.0)
        std = np.sqrt(var + float(eps))
        std = np.maximum(std, float(eps))
        return std.astype(np.float64)


def vectorize_feature_map(feat_chw: np.ndarray) -> np.ndarray:
    """
    将 [C, H, W] 特征图展平为 [N, C] 向量矩阵。
    English: [C, H, W] [N, C] .
    """
    feat = np.asarray(feat_chw, dtype=np.float32)
    c = int(feat.shape[0])
    return feat.reshape(c, -1).T.astype(np.float32)


def normalize_vectors(X: np.ndarray, center: np.ndarray, scale: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    逻辑：
    English: Logic:
    1. 对 [N, C] 向量矩阵执行按通道 z-score 归一化；
    English: 1. [N, C] z-score ;
    2. 该函数同时服务于离线先验构建与后续画图脚本；
    English: 2. build;
    3. 不在这里做任何可学习仿射变换，保持“固定先验归一化”语义。
    English: 3. , “”.
    """
    X = np.asarray(X, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32).reshape(1, -1)
    scale = np.asarray(scale, dtype=np.float32).reshape(1, -1)
    return ((X - center) / (scale + float(eps))).astype(np.float32)


def normalize_feature_map_channels(feat_chw: np.ndarray, center: np.ndarray, scale: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    对 [C, H, W] 特征图执行逐通道固定归一化。
    English: [C, H, W] .
    """
    feat = np.asarray(feat_chw, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32).reshape(-1, 1, 1)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1, 1, 1)
    return ((feat - center) / (scale + float(eps))).astype(np.float32)


def prepare_band_feature_map(
    pyramid: Dict[str, np.ndarray],
    band_name: str,
    crop_size_map: Dict[str, Optional[int]]
) -> np.ndarray:
    """
    逻辑：
    English: Logic:
    1. 统一管理“先金字塔，再按频层中心裁切”的入口；
    English: 1. “, ”;
    2. 之前旧版脚本虽然定义了 crop 参数，但主流程未真正使用；
    English: 2. crop parameter, ;
    3. 当前版本在这里显式接上，避免后续再次遗漏。
    English: 3. currentexplicit, avoid.
    """
    if band_name not in pyramid:
        raise KeyError(f"金字塔结果中缺少频层: {band_name}")
    crop_size = crop_size_map.get(band_name, None)
    return center_crop_feature_map(pyramid[band_name], crop_size=crop_size)


def build_band_normalization_priors(
    scanner,
    records,
    max_samples,
    crop_size_map,
    norm_eps: float
):
    """
    第一遍扫描：统计各频层在向量空间中的逐通道 mean / std。
    English: std.
    """
    band_norm_stats = {
        "high1": RunningChannelNormStats(channel_dim=len(WAVELENGTHS)),
        "high2": RunningChannelNormStats(channel_dim=len(WAVELENGTHS)),
        "high3": RunningChannelNormStats(channel_dim=len(WAVELENGTHS)),
        "low": RunningChannelNormStats(channel_dim=len(WAVELENGTHS))
    }

    processed = 0
    for item in scanner.iter_calibrated_images(records, max_samples=max_samples):
        pyramid = build_laplacian_pyramid(item["image"], num_levels=3)

        for band_name in ["high1", "high2", "high3", "low"]:
            feat = prepare_band_feature_map(pyramid, band_name, crop_size_map)
            band_norm_stats[band_name].update(vectorize_feature_map(feat))

        processed += 1
        if processed % 20 == 0:
            print(f">> [Pass-1/2] 已累计归一化统计样本: {processed}")

    if processed == 0:
        raise RuntimeError("第一遍扫描未成功处理任何样本，无法构造通道归一化先验。")

    band_norm_priors = {}
    for band_name, stats in band_norm_stats.items():
        center = stats.get_mean().astype(np.float32)
        scale = stats.get_std(eps=norm_eps).astype(np.float32)
        band_norm_priors[band_name] = {
            "norm_method": "zscore_mean_std",
            "norm_center": center,
            "norm_scale": scale,
            "norm_eps": float(norm_eps),
            "vector_count": int(stats.count)
        }

    return band_norm_priors
class RunningPCAStats:
    """
    逻辑：
    English: Logic:
    1. 仅累计 PCA 所需的充分统计量：样本数、按通道求和、按通道外积和；
    English: 1. PCA : sample, , ;
    2. 对于本工程的 8 通道输入，最终只需要 8×8 协方差矩阵，
    English: 2. 8 Input, 8×8 ,.
       因此完全没有必要把数亿行向量整体拼接进内存；
       English: ;
    3. 这样可显著降低峰值内存占用，尤其适合 low 频层全量保留的情况。
    English: 3. , low .
    """
    def __init__(self, channel_dim: int):
        """
        初始化 PCA 充分统计量容器。
        English: PCA .

        输入:
        English: Input:
            channel_dim: PCA 向量通道数，本工程固定为 8。
            English: channel_dim: PCA , 8.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        self.channel_dim = int(channel_dim)
        self.count = 0
        self.sum_vec = np.zeros((self.channel_dim,), dtype=np.float64)
        self.sum_outer = np.zeros((self.channel_dim, self.channel_dim), dtype=np.float64)

    def update(self, X: np.ndarray):
        """
        累计一批进入 PCA 的向量。
        English: PCA .

        输入:
        English: Input:
            X: [N, C] 向量矩阵，高频层为筛选后的结构向量，低频层为全量向量。
            English: X: [N, C] , , .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if X is None:
            return
        X = np.asarray(X)
        if X.size == 0:
            return
        if X.ndim != 2:
            raise ValueError(f"RunningPCAStats.update 要求二维输入 [N, C]，当前维度为 {X.ndim}")
        if X.shape[1] != self.channel_dim:
            raise ValueError(
                f"通道数不一致，期望 {self.channel_dim}，当前为 {X.shape[1]}"
            )

        X32 = X.astype(np.float32, copy=False)
        self.count += int(X32.shape[0])
        self.sum_vec += X32.sum(axis=0, dtype=np.float64)
        self.sum_outer += np.einsum('ni,nj->ij', X32, X32, dtype=np.float64)

    def get_mean(self) -> np.ndarray:
        """
        从累计充分统计量计算 PCA 均值向量。
        English: calculate PCA .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if self.count <= 0:
            raise ValueError("当前没有任何样本，无法计算均值")
        return (self.sum_vec / float(self.count)).astype(np.float64)

    def get_covariance(self, eps: float = 1e-8) -> np.ndarray:
        """
        从累计充分统计量计算协方差矩阵。
        English: calculate.

        输入:
        English: Input:
            eps: 对角线数值稳定项。
            English: eps: .
        输出:
        English: Output:
            [C, C] 对称协方差矩阵。
            English: [C, C] .

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        if self.count < 2:
            raise ValueError(f"PCA 样本数不足，当前 N={self.count}")

        mu = self.get_mean()
        centered_outer = self.sum_outer - float(self.count) * np.outer(mu, mu)
        cov = centered_outer / max(self.count - 1, 1)
        cov = 0.5 * (cov + cov.T)
        cov += np.eye(self.channel_dim, dtype=np.float64) * eps
        return cov.astype(np.float64)


def fit_pca_from_running_stats(stats: RunningPCAStats, eta=0.99, max_components=None, eps=1e-8):
    """
    基于流式累计得到的充分统计量进行 PCA。
    English: PCA.
    """
    if not isinstance(stats, RunningPCAStats):
        raise TypeError("stats 必须为 RunningPCAStats 实例")

    n = int(stats.count)
    c = int(stats.channel_dim)
    if n < 2:
        raise ValueError(f"PCA 样本数不足，当前 N={n}")

    mu = stats.get_mean()
    cov = stats.get_covariance(eps=eps)

    eigvals_full, eigvecs_full = np.linalg.eigh(cov)
    order = np.argsort(eigvals_full)[::-1]
    eigvals_full = eigvals_full[order]
    eigvecs_full = eigvecs_full[:, order]

    eigvals_full = np.clip(eigvals_full, a_min=0.0, a_max=None)
    total = np.sum(eigvals_full) + eps
    explained_ratio_full = eigvals_full / total
    cumulative_ratio_full = np.cumsum(explained_ratio_full)

    k = int(np.searchsorted(cumulative_ratio_full, eta) + 1)
    k = max(1, min(k, c))
    if max_components is not None:
        k = min(k, int(max_components))

    components = eigvecs_full[:, :k].T.astype(np.float32)       # [K, C]
    eigvals = eigvals_full[:k].astype(np.float32)
    explained_ratio = explained_ratio_full[:k].astype(np.float32)
    cumulative_ratio = cumulative_ratio_full[:k].astype(np.float32)

    mean = mu.astype(np.float32)
    weight = components.copy()                                  # [K, C]
    bias = (-components @ mean).astype(np.float32)              # [K]

    component_loadings = components.copy()                      # [K, C]

    return {
        "mean": mean,
        "components": components,
        "eigvals": eigvals,
        "explained_variance_ratio": explained_ratio,
        "cumulative_explained_variance": cumulative_ratio,
        "eigvals_full": eigvals_full.astype(np.float32),
        "explained_variance_ratio_full": explained_ratio_full.astype(np.float32),
        "cumulative_explained_variance_full": cumulative_ratio_full.astype(np.float32),
        "selected_k": int(k),
        "full_dim": int(c),
        "weight": weight,
        "bias": bias,
        "covariance": cov.astype(np.float32),
        "component_loadings": component_loadings.astype(np.float32),
        "sample_count": int(n)
    }


def fit_pca_from_vectors(X, eta=0.99, max_components=None, eps=1e-8):
    """
    兼容旧接口：当确实已有 [N, C] 全量向量矩阵时，仍可直接调用；
    English: compatible: [N, C] , ;
    内部会自动转为流式统计逻辑，避免 np.cov 产生巨型临时副本。
    English: Logic, avoid np.cov .
    """
    if X.ndim != 2:
        raise ValueError("X 必须为二维 [N, C]")

    stats = RunningPCAStats(channel_dim=int(X.shape[1]))
    stats.update(X)
    return fit_pca_from_running_stats(stats, eta=eta, max_components=max_components, eps=eps)


# ================= 6. 训练期 Fold 内先验构建 =================
# EN: ================= 6. train Fold inside first build =================.
# 逻辑：
# EN: Logic:
# 1. 正式交叉验证训练时，PCA / 通道归一化先验只能由当前 Fold 的 Train 子集估计；
# EN: formalcross-validationtrain when, PCA / channel normalization first only can by current Fold Train estimate;
# 2. Validation / Test 样本不得参与 norm_center、norm_scale、结构向量筛选、PCA 均值、
# EN: Validation / Test samplemust notparticipate in norm_center, norm_scale, structural-vector selection, PCA mean,.
#    协方差、载荷或主成分数量估计，避免数据泄露；
# EN: , or number of principal componentsestimate, avoiddata;
# 3. 这里复用离线构建脚本的两遍算法，只把样本来源替换为训练期 Dataset + train_indices；
# EN: herereuse build, only sample source as train Dataset + train_indices;
# 4. 最近修改时间：2026-06-17；作者：GG。
# EN: Last modified: 2026-06-17; Author: GG.

SUPPORTED_TRAINING_PRIOR_STRUCTURES = {
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


def normalize_training_prior_structure(structure) -> Tuple[str, ...]:
    """
    将训练菜单结构标签转换为需要估计先验的频层名称。
    English: trainingmenulabelname.

    输入:
    English: Input:
        structure: 菜单中的 Full / H2H3Low / H3Low / LowOnly 等结构标签。
        English: H2H3Low / H3Low / LowOnly 等结构标签.
    输出:
    English: Output:
        按模型实际命名的频层元组。
        English: model.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    if structure is None or structure == "":
        return SUPPORTED_TRAINING_PRIOR_STRUCTURES["full"]
    if isinstance(structure, (list, tuple)):
        bands = tuple(str(item).strip().lower() for item in structure if str(item).strip())
        if bands in set(SUPPORTED_TRAINING_PRIOR_STRUCTURES.values()):
            return bands
        raise ValueError(f"不支持的 MFPC-HFNet 训练期先验结构频层组合: {bands}。")

    key = str(structure).strip().lower()
    key = key.replace(" ", "")
    key = key.replace("-", "_")
    key = key.replace("/", "+")
    if key in SUPPORTED_TRAINING_PRIOR_STRUCTURES:
        return SUPPORTED_TRAINING_PRIOR_STRUCTURES[key]

    parts = tuple(part for part in key.split("+") if part)
    alias = {"h1": "high1", "h2": "high2", "h3": "high3"}
    bands = tuple(alias.get(part, part) for part in parts)
    if bands in set(SUPPORTED_TRAINING_PRIOR_STRUCTURES.values()):
        return bands
    raise ValueError(f"不支持的 MFPC-HFNet 训练期先验结构: {structure!r}。")


def build_structure_laplacian_pyramid_for_training_prior(img_chw: np.ndarray, structure) -> Dict[str, np.ndarray]:
    """
    按菜单结构构建与模型 forward 对齐的拉普拉斯频层。
    English: menubuildmodel forward .

    设计说明:
    English: Design note:
    - Full 输入 1024 时输出 high1/high2/high3/low；
    English: - Full Input 1024 Output high1/high2/high3/low;
    - H2H3Low 输入 512 时第一道高频响应命名为 high2，而不是 high1；
    English: - H2H3Low Input 512 high2, high1;
    - H3Low 输入 256 时第一道高频响应命名为 high3；
    English: - H3Low Input 256 high3;
    - LowOnly 输入 128 时仅估计 low；
    English: - LowOnly Input 128 low;
    - 最近修改时间：2026-06-17；作者：GG。
    English: - Last modified: 2026-06-17; Author: GG.
    """

    bands = normalize_training_prior_structure(structure)
    current = np.asarray(img_chw, dtype=np.float32)
    if current.ndim != 3:
        raise ValueError(f"训练期先验图像必须为 [C, H, W]，当前 shape={current.shape}。")
    if int(current.shape[0]) != len(WAVELENGTHS):
        raise ValueError(f"训练期先验图像通道数必须为 {len(WAVELENGTHS)}，当前为 {current.shape[0]}。")

    pyramid: Dict[str, np.ndarray] = {}
    high_bands = [band for band in bands if band != "low"]
    if high_bands:
        kernel = build_gaussian_kernel()
        for band_name in high_bands:
            down = pyramid_down(current, kernel)
            up = pyramid_up(down, current.shape[1:])
            pyramid[band_name] = (current - up).astype(np.float32)
            current = down.astype(np.float32)
    pyramid["low"] = current.astype(np.float32)
    return pyramid


def tensor_or_array_to_image_chw(value) -> np.ndarray:
    """
    将 Dataset 返回的 image 转成 float32 numpy [C, H, W]。
    English: Dataset return image float32 numpy [C, H, W].

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    if isinstance(value, torch.Tensor):
        image = value.detach().cpu().numpy()
    else:
        image = np.asarray(value)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"image 必须为三维张量，当前 shape={image.shape}。")
    if image.shape[0] == len(WAVELENGTHS):
        return image
    if image.shape[-1] == len(WAVELENGTHS):
        return np.transpose(image, (2, 0, 1)).astype(np.float32)
    raise ValueError(f"无法识别 image 通道维，当前 shape={image.shape}。")


def iter_dataset_training_images(dataset, train_indices, max_samples=None):
    """
    从训练期 Dataset 按 train_indices 二次遍历图像样本。
    English: training Dataset train_indices imagesample.

    输入:
    English: Input:
        dataset: 已由 Train_core 按当前 ModelSpec 构建完成的 Dataset。
        English: dataset: Train_core current ModelSpec build Dataset.
        train_indices: 当前 Fold 的训练索引，只允许包含 Train 子集。
        English: train_indices: current Fold training, Train .
        max_samples: 可选调试上限；正式训练默认为 None。
        English: max_samples: optional; trainingdefault None.
    输出:
    English: Output:
        逐个 yield 包含 image、sample_name、core_id、folder_path 的字典。
        English: yield image, sample_name, core_id, folder_path dictionary.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    records = getattr(dataset, "data_cache", [])
    count = 0
    for dataset_index in [int(item) for item in train_indices]:
        item = dataset[dataset_index]
        if "image" not in item:
            raise KeyError("训练期先验构建需要 image 输入，但当前 Dataset 样本未返回 image。")
        record = records[dataset_index] if 0 <= dataset_index < len(records) else {}
        yield {
            "dataset_index": int(dataset_index),
            "image": tensor_or_array_to_image_chw(item["image"]),
            "sample_name": item.get("sample_name", record.get("sample_name", str(dataset_index))),
            "core_id": item.get("core_id", record.get("core_id", "")),
            "folder_path": item.get("folder_path", record.get("folder_path", "")),
        }
        count += 1
        if max_samples is not None and count >= int(max_samples):
            break


def build_training_band_normalization_priors(
    dataset,
    train_indices,
    structure,
    crop_size_map,
    norm_eps: float,
    max_samples=None,
):
    """
    基于当前 Fold Train 子集统计各启用频层的通道归一化参数。
    English: current Fold Train parameter.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    active_bands = normalize_training_prior_structure(structure)
    band_norm_stats = {
        band_name: RunningChannelNormStats(channel_dim=len(WAVELENGTHS))
        for band_name in active_bands
    }

    processed = 0
    for item in iter_dataset_training_images(dataset, train_indices, max_samples=max_samples):
        pyramid = build_structure_laplacian_pyramid_for_training_prior(item["image"], structure)
        for band_name in active_bands:
            feat = prepare_band_feature_map(pyramid, band_name, crop_size_map)
            band_norm_stats[band_name].update(vectorize_feature_map(feat))
        processed += 1
        if processed % 20 == 0:
            print(f">> [Fold PCA Pass-1/2] 已累计训练样本: {processed}")

    if processed == 0:
        raise RuntimeError("Fold 训练集先验第一遍未处理任何样本。")

    band_norm_priors = {}
    for band_name, stats in band_norm_stats.items():
        band_norm_priors[band_name] = {
            "norm_method": "zscore_mean_std",
            "norm_center": stats.get_mean().astype(np.float32),
            "norm_scale": stats.get_std(eps=norm_eps).astype(np.float32),
            "norm_eps": float(norm_eps),
            "vector_count": int(stats.count),
        }
    return band_norm_priors, int(processed)


def clone_prior_band_for_inactive_band(band_prior: dict) -> dict:
    """
    为模型校验所需但当前结构未实例化的频层复制一个占位先验。
    English: modelvalidationcurrent.

    说明:
    English: :
        Model_MFPCHFNet 当前要求先验字典包含 high1/high2/high3/low 四个键；
        English: Model_MFPCHFNet currentdictionary high1/high2/high3/low ;
        结构消融中未启用的频层不会被实例化或参与 forward，因此该占位只用于保持字典合同完整。
        English: forward, dictionary.

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    cloned = {}
    for key, value in band_prior.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    cloned["inactive_band_placeholder"] = True
    return cloned


def build_pca_priors_from_training_dataset(
    *,
    dataset,
    train_indices,
    output_dir,
    structure="high1+high2+high3+low",
    metadata=None,
    max_samples=None,
    eta_high=DEFAULT_ETA_HIGH,
    eta_low=DEFAULT_ETA_LOW,
    max_components=DEFAULT_MAX_COMPONENTS,
    snr_db=DEFAULT_SNR_DB,
    chi2_quantile=DEFAULT_CHI2_QUANTILE,
    fdr_q=DEFAULT_FDR_Q,
    selection_min_keep_ratio=DEFAULT_SELECTION_MIN_KEEP_RATIO,
    selection_min_keep_channel_multiplier=DEFAULT_SELECTION_MIN_KEEP_CHANNEL_MULTIPLIER,
    selection_min_keep_abs=DEFAULT_SELECTION_MIN_KEEP_ABS,
    background_min_count=DEFAULT_BACKGROUND_MIN_COUNT,
    background_min_channel_multiplier=DEFAULT_BACKGROUND_MIN_CHANNEL_MULTIPLIER,
    background_fallback_base=DEFAULT_BACKGROUND_FALLBACK_BASE,
    background_fallback_channel_multiplier=DEFAULT_BACKGROUND_FALLBACK_CHANNEL_MULTIPLIER,
    max_background_for_cov=DEFAULT_MAX_BACKGROUND_FOR_COV,
    cov_shrinkage_alpha=DEFAULT_COV_SHRINKAGE_ALPHA,
    cov_shrinkage_eps=DEFAULT_COV_SHRINKAGE_EPS,
    seed=42,
    crop_high1=DEFAULT_CROP_HIGH1,
    crop_high2=DEFAULT_CROP_HIGH2,
    crop_high3=DEFAULT_CROP_HIGH3,
    crop_low=DEFAULT_CROP_LOW,
    norm_eps=DEFAULT_NORM_EPS,
):
    """
    使用当前 Fold 的训练集重构 MFPC-HFNet PCA/归一化先验。
    English: current Fold training MFPC-HFNet PCA/.

    输出:
    English: Output:
        返回包含 `priors_path`、`summary_path` 和关键追溯字段的字典，同时写出:
        - pca_priors_train_only.pt
        - pca_priors_train_only_summary.json

    最近修改时间：2026-06-17；作者：GG。
    English: Last modified: 2026-06-17; Author: GG.
    """

    set_seed(seed)
    output_dir = str(output_dir)
    ensure_dir(output_dir)
    train_indices = [int(item) for item in train_indices]
    if not train_indices:
        raise ValueError("Fold 训练集索引为空，无法重构 PCA 先验。")

    active_bands = normalize_training_prior_structure(structure)
    crop_size_map = {
        "high1": crop_high1,
        "high2": crop_high2,
        "high3": crop_high3,
        "low": crop_low,
    }
    metadata = dict(metadata or {})

    print(">> [Fold PCA Pass-1/2] 仅使用当前 Fold 训练集统计通道归一化参数")
    band_norm_priors, pass1_processed = build_training_band_normalization_priors(
        dataset=dataset,
        train_indices=train_indices,
        structure=structure,
        crop_size_map=crop_size_map,
        norm_eps=norm_eps,
        max_samples=max_samples,
    )

    band_stats = {
        band_name: RunningPCAStats(channel_dim=len(WAVELENGTHS))
        for band_name in active_bands
    }
    sample_level_rows = []
    processed = 0

    print(">> [Fold PCA Pass-2/2] 仅使用当前 Fold 训练集执行结构筛选与 PCA 统计")
    for item in iter_dataset_training_images(dataset, train_indices, max_samples=max_samples):
        img = item["image"]
        pyramid = build_structure_laplacian_pyramid_for_training_prior(img, structure)

        normalized_features = {}
        for band_name in active_bands:
            normalized_features[band_name] = normalize_feature_map_channels(
                prepare_band_feature_map(pyramid, band_name, crop_size_map),
                band_norm_priors[band_name]["norm_center"],
                band_norm_priors[band_name]["norm_scale"],
                eps=band_norm_priors[band_name]["norm_eps"],
            )

        row = {
            "dataset_index": int(item["dataset_index"]),
            "sample_name": item["sample_name"],
            "core_id": item["core_id"],
            "folder_path": item["folder_path"],
            "orig_h": int(img.shape[1]),
            "orig_w": int(img.shape[2]),
        }
        for band_name in active_bands:
            if band_name == "low":
                vectors = vectorize_feature_map(normalized_features[band_name])
                band_stats[band_name].update(vectors)
                row[f"{band_name}_total_vectors"] = int(vectors.shape[0])
                row[f"{band_name}_kept_count"] = int(vectors.shape[0])
                row[f"{band_name}_kept_ratio"] = 1.0
                continue

            vectors, info, _, _ = select_structural_vectors(
                normalized_features[band_name],
                snr_db=snr_db,
                chi2_quantile=chi2_quantile,
                fdr_q=fdr_q,
                min_keep_ratio=selection_min_keep_ratio,
                min_keep_channel_multiplier=selection_min_keep_channel_multiplier,
                min_keep_abs=selection_min_keep_abs,
                background_min_count=background_min_count,
                background_min_channel_multiplier=background_min_channel_multiplier,
                background_fallback_base=background_fallback_base,
                background_fallback_channel_multiplier=background_fallback_channel_multiplier,
                max_background_for_cov=max_background_for_cov,
                cov_shrinkage_alpha=cov_shrinkage_alpha,
                cov_shrinkage_eps=cov_shrinkage_eps,
            )
            band_stats[band_name].update(vectors)
            row[f"{band_name}_total_vectors"] = int(info["total_vectors"])
            row[f"{band_name}_kept_count"] = int(info["kept_count"])
            row[f"{band_name}_kept_ratio"] = float(info["kept_ratio"])

        sample_level_rows.append(row)
        processed += 1
        if processed % 20 == 0:
            print(f">> [Fold PCA Pass-2/2] 已处理训练样本: {processed}")

    if processed == 0:
        raise RuntimeError("Fold 训练集先验第二遍未处理任何样本。")

    pca_results = {}
    for band_name in active_bands:
        pca_results[band_name] = fit_pca_from_running_stats(
            band_stats[band_name],
            eta=eta_low if band_name == "low" else eta_high,
            max_components=max_components,
        )

    sample_stats_df = pd.DataFrame(sample_level_rows)
    pcase_vector_stats = {}
    for band_name in active_bands:
        total_col = f"{band_name}_total_vectors"
        kept_col = f"{band_name}_kept_count"
        total_candidate_vectors = int(sample_stats_df[total_col].sum())
        effective_vector_count = int(sample_stats_df[kept_col].sum())
        effective_feature_vector_ratio = (
            float(effective_vector_count) / float(total_candidate_vectors)
            if total_candidate_vectors > 0 else 0.0
        )
        if band_name == "low":
            effective_feature_vector_ratio = 1.0
        pcase_vector_stats[band_name] = {
            "total_candidate_vectors": total_candidate_vectors,
            "effective_vector_count": effective_vector_count,
            "effective_feature_vector_ratio": float(effective_feature_vector_ratio),
            "feature_vector_ratio_source": (
                "fixed_low_no_vector_screening"
                if band_name == "low" else "fold_train_only_structural_vector_screening"
            ),
        }

    priors = {
        "meta": {
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prior_build_scope": "fold_train_only",
            "leakage_guard": "validation_and_test_samples_excluded_from_prior_estimation",
            "processed_samples": int(processed),
            "pass1_processed_samples": int(pass1_processed),
            "train_indices_count": int(len(train_indices)),
            "active_structure": str(structure),
            "active_bands": list(active_bands),
            "inactive_bands": [band for band in ["high1", "high2", "high3", "low"] if band not in active_bands],
            "snr_db": float(snr_db),
            "chi2_quantile": float(chi2_quantile),
            "fdr_q": float(fdr_q),
            "selection_min_keep_ratio": float(selection_min_keep_ratio),
            "selection_min_keep_channel_multiplier": int(selection_min_keep_channel_multiplier),
            "selection_min_keep_abs": int(selection_min_keep_abs),
            "background_min_count": int(background_min_count),
            "background_min_channel_multiplier": int(background_min_channel_multiplier),
            "background_fallback_base": int(background_fallback_base),
            "background_fallback_channel_multiplier": int(background_fallback_channel_multiplier),
            "max_background_for_cov": int(max_background_for_cov),
            "cov_shrinkage_alpha": float(cov_shrinkage_alpha),
            "cov_shrinkage_eps": float(cov_shrinkage_eps),
            "eta_high": float(eta_high),
            "eta_low": float(eta_low),
            "max_components": None if max_components is None else int(max_components),
            "crop_high1": None if crop_high1 is None else int(crop_high1),
            "crop_high2": None if crop_high2 is None else int(crop_high2),
            "crop_high3": None if crop_high3 is None else int(crop_high3),
            "crop_low": None if crop_low is None else int(crop_low),
            "wavelengths_nm": WAVELENGTHS_NUM.tolist(),
            "channel_norm_method": "zscore_mean_std",
            "channel_norm_eps": float(norm_eps),
            "pcase_vector_stats": pcase_vector_stats,
            **metadata,
        }
    }

    for band_name in active_bands:
        result = pca_results[band_name]
        priors[band_name] = {
            "mean": torch.from_numpy(result["mean"]),
            "components": torch.from_numpy(result["components"]),
            "eigvals": torch.from_numpy(result["eigvals"]),
            "explained_variance_ratio": torch.from_numpy(result["explained_variance_ratio"]),
            "cumulative_explained_variance": torch.from_numpy(result["cumulative_explained_variance"]),
            "eigvals_full": torch.from_numpy(result["eigvals_full"]),
            "explained_variance_ratio_full": torch.from_numpy(result["explained_variance_ratio_full"]),
            "cumulative_explained_variance_full": torch.from_numpy(result["cumulative_explained_variance_full"]),
            "weight": torch.from_numpy(result["weight"]),
            "bias": torch.from_numpy(result["bias"]),
            "covariance": torch.from_numpy(result["covariance"]),
            "component_loadings": torch.from_numpy(result["component_loadings"]),
            "norm_method": band_norm_priors[band_name]["norm_method"],
            "norm_center": torch.from_numpy(band_norm_priors[band_name]["norm_center"]),
            "norm_scale": torch.from_numpy(band_norm_priors[band_name]["norm_scale"]),
            "norm_eps": float(band_norm_priors[band_name]["norm_eps"]),
            "selected_k": int(result["selected_k"]),
            "full_dim": int(result["full_dim"]),
            "total_candidate_vectors": int(pcase_vector_stats[band_name]["total_candidate_vectors"]),
            "effective_vector_count": int(pcase_vector_stats[band_name]["effective_vector_count"]),
            "effective_feature_vector_ratio": float(pcase_vector_stats[band_name]["effective_feature_vector_ratio"]),
            "feature_vector_ratio_source": pcase_vector_stats[band_name]["feature_vector_ratio_source"],
        }

    placeholder_source = active_bands[0]
    for band_name in ["high1", "high2", "high3", "low"]:
        if band_name not in priors:
            priors[band_name] = clone_prior_band_for_inactive_band(priors[placeholder_source])
            priors[band_name]["placeholder_source_band"] = placeholder_source

    priors_path = os.path.join(output_dir, "pca_priors_train_only.pt")
    summary_path = os.path.join(output_dir, "pca_priors_train_only_summary.json")
    torch.save(priors, priors_path)

    summary = {
        "priors_path": priors_path,
        "summary_path": summary_path,
        "prior_build_scope": "fold_train_only",
        "leakage_guard": "validation_and_test_samples_excluded_from_prior_estimation",
        "processed_samples": int(processed),
        "train_indices_count": int(len(train_indices)),
        "active_structure": str(structure),
        "active_bands": list(active_bands),
        "inactive_bands": priors["meta"]["inactive_bands"],
        "pcase_vector_stats": pcase_vector_stats,
        **metadata,
    }
    save_json(summary, summary_path)
    print(f">> [Fold PCA] train-only priors saved: {priors_path}")
    return summary


# ================= 6. 跨频层差异分析 =================
# EN: ================= 6. frequency band =================.
# 逻辑：
# EN: Logic:
# 1. 为证明“先拉普拉斯分频再 PCA”是合理的，需要保存跨频层差异数值；
# EN: as " PCA" is, need need save frequency band value;
# 2. 这里保存：
# EN: heresave:
#    - 均值向量 L2 距离
# EN: mean amount L2.
#    - 前 K 子空间主角
# EN: before K.
#    - 投影矩阵差异 Frobenius 范数
# EN: Frobenius number.
#    - 主成分载荷余弦相似度
# EN: principal components degree.
# 3. 后续可直接据此画热图或表格。
# EN: later can directly data image or table.
def l2_distance(a, b):
    """
    计算两个向量之间的 L2 距离。
    English: calculate L2 .

    输入:
    English: Input:
        a / b: 同维度向量。
        English: b: 同维度向量.
    输出:
    English: Output:
        Python float，便于 JSON/CSV 直接保存。
        English: Python float, JSON/CSV save.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    return float(np.linalg.norm(a - b))


def cosine_similarity_matrix(A, B, eps=1e-8):
    """
    A: [Ka, C]
    B: [Kb, C]
    return [Ka, Kb]
    """
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return (A_n @ B_n.T).astype(np.float32)


def principal_angles_deg(Ua, Ub, k=None):
    """
    Ua: [Ka, C]
    Ub: [Kb, C]
    每行是一个基向量，已近似正交
    English: ,.
    """
    Qa = Ua[:k].T if k is not None else Ua.T
    Qb = Ub[:k].T if k is not None else Ub.T

    # [C, k]
    qa, _ = np.linalg.qr(Qa)
    qb, _ = np.linalg.qr(Qb)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s) * 180.0 / np.pi
    return angles.astype(np.float32)


def projection_matrix(U, k=None):
    """
    U: [K, C]
    """
    Uk = U[:k] if k is not None else U
    return (Uk.T @ Uk).astype(np.float32)


def build_cross_band_analysis(pca_results: Dict[str, dict], compare_topk=3):
    """
    构建 high1/high2/high3/low 之间的 PCA 子空间差异分析。
    English: build high1/high2/high3/low PCA .

    输入:
    English: Input:
        pca_results: 各频层 PCA 拟合结果字典。
        English: pca_results: PCA resultdictionary.
        compare_topk: 比较前 K 个主成分子空间。
        English: compare_topk: K .
    输出:
    English: Output:
        包含均值距离、主角、投影矩阵距离和载荷余弦相似度的结构化字典。
        English: , , dictionary.

    用途:
    English: :
        这些数值用于后续论文作图或表格，证明不同频层 PCA 空间并非简单重复。
        English: , PCA .
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
    """
    band_names = ["high1", "high2", "high3", "low"]
    out = {
        "mean_l2_distance": {},
        "subspace_principal_angles_deg": {},
        "projection_frobenius_distance": {},
        "component_cosine_similarity_topk": {}
    }

    for i, a in enumerate(band_names):
        for j, b in enumerate(band_names):
            key = f"{a}__vs__{b}"

            mean_a = pca_results[a]["mean"]
            mean_b = pca_results[b]["mean"]
            out["mean_l2_distance"][key] = l2_distance(mean_a, mean_b)

            Ua = pca_results[a]["components"]
            Ub = pca_results[b]["components"]

            angles = principal_angles_deg(Ua, Ub, k=compare_topk)
            out["subspace_principal_angles_deg"][key] = angles.tolist()

            Pa = projection_matrix(Ua, k=compare_topk)
            Pb = projection_matrix(Ub, k=compare_topk)
            proj_dist = np.linalg.norm(Pa - Pb, ord="fro")
            out["projection_frobenius_distance"][key] = float(proj_dist)

            cos_sim = cosine_similarity_matrix(Ua[:compare_topk], Ub[:compare_topk])
            out["component_cosine_similarity_topk"][key] = cos_sim.tolist()

    return out


# ================= 7. 数值缓存策略 =================
# EN: ================= 7. valuecache =================.
# 逻辑：
# EN: Logic:
# 1. 为避免缓存过大，只对代表样本保存较完整的中间数值；
# EN: as avoidcache large, only for table samplesave complete in value;
# 2. 这些缓存只保存 npz 数值，不直接输出图片；
# EN: cacheonlysave npz value, not directly image;
# 3. 后续你可以自由读取并画原图/频层图/掩膜/PC 图。
# EN: later can by read and image /frequency band image / /PC image.
def choose_visual_cache_indices(total_count, max_visual_samples, seed=42):
    """
    选择需要保存完整数值缓存的代表样本下标。
    English: selectsavecachesample.

    输入:
    English: Input:
        total_count: 本轮实际处理样本数。
        English: total_count: sample.
        max_visual_samples: 最多缓存多少个样本。
        English: max_visual_samples: cachesample.
        seed: 抽样随机种子。
        English: seed: .
    输出:
    English: Output:
        样本下标集合。
        English: sample.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    if total_count <= max_visual_samples:
        return set(range(total_count))
    rng = np.random.default_rng(seed)
    idx = rng.choice(total_count, size=max_visual_samples, replace=False)
    return set(idx.tolist())


def project_feature_map_with_pca(feat_chw, mean, components, topk=3):
    """
    feat_chw: [C, H, W]
    mean: [C]
    components: [K, C]
    return: [topk, H, W]
    """
    c, h, w = feat_chw.shape
    X = feat_chw.reshape(c, -1).T.astype(np.float32)  # [N, C]
    Xc = X - mean[None, :]
    k = min(topk, components.shape[0])
    Y = Xc @ components[:k].T
    return Y.T.reshape(k, h, w).astype(np.float32)


def save_numeric_visual_cache(
    save_dir,
    item,
    pyramid,
    aux_maps,
    pca_maps=None
):
    """
    只保存数值缓存，不保存图片
    English: savecache, save.
    """
    sample_name = item["sample_name"]
    out_path = os.path.join(save_dir, f"{sample_name}.npz")

    save_dict = {
        "sample_name": np.array(sample_name),
        "core_id": np.array(item["core_id"]),
        "wavelengths_nm": WAVELENGTHS_NUM.astype(np.float32),

        "image_calibrated": item["image"].astype(np.float32),

        "high1": pyramid["high1"].astype(np.float32),
        "high2": pyramid["high2"].astype(np.float32),
        "high3": pyramid["high3"].astype(np.float32),
        "low": pyramid["low"].astype(np.float32),
    }

    for band in ["high1", "high2", "high3"]:
        save_dict[f"{band}_energy_map"] = aux_maps[band]["energy_map"].astype(np.float32)
        save_dict[f"{band}_d2_map"] = aux_maps[band]["d2_map"].astype(np.float32)
        save_dict[f"{band}_pval_map"] = aux_maps[band]["pval_map"].astype(np.float32)
        save_dict[f"{band}_structure_mask"] = aux_maps[band]["structure_mask"].astype(np.uint8)
        save_dict[f"{band}_robust_mean"] = aux_maps[band]["robust_mean"].astype(np.float32)
        save_dict[f"{band}_covariance"] = aux_maps[band]["covariance"].astype(np.float32)

    if pca_maps is not None:
        for band, arr in pca_maps.items():
            save_dict[f"{band}_pc_maps"] = arr.astype(np.float32)

    np.savez_compressed(out_path, **save_dict)


# ================= 8. 结果打包与表格输出 =================
# EN: ================= 8. result result and table =================.
# 逻辑：
# EN: Logic:
# 1. 将多种分析结果分别保存为 pt / csv / json / npz；
# EN: more result result save as pt / csv / json / npz;
# 2. 尽量保证“模型构造”和“论文分析”都能直接读取；
# EN: amount ensure"" and "" all can directlyread;
# 3. 所有输出均为数值数据。
# EN: as valuedata.
def save_sample_stats_csv(rows: List[dict], path: str):
    """
    保存样本级结构向量筛选统计表。
    English: savesample.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_band_summary_csv(rows: List[dict], path: str):
    """
    保存频层级 PCA 与筛选汇总表。
    English: save PCA .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_scree_csv(pca_results: Dict[str, dict], path: str):
    """
    保存每个频层 PCA 全谱 scree 数据。
    English: save PCA scree .

    输入:
    English: Input:
        pca_results: 各频层 PCA 结果。
        English: pca_results: PCA result.
        path: 输出 CSV 路径。
        English: path: Output CSV path.
    输出字段:
    English: Outputfield:
        eigval、explained_variance_ratio 和 cumulative_explained_variance。
        English: eigval, explained_variance_ratio cumulative_explained_variance.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    rows = []
    for band, result in pca_results.items():
        eigvals_full = result["eigvals_full"]
        ratio_full = result["explained_variance_ratio_full"]
        cumsum_full = result["cumulative_explained_variance_full"]

        for idx in range(len(eigvals_full)):
            rows.append({
                "band": band,
                "component_index_1based": idx + 1,
                "eigval": float(eigvals_full[idx]),
                "explained_variance_ratio": float(ratio_full[idx]),
                "cumulative_explained_variance": float(cumsum_full[idx])
            })

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_component_loadings_csv(pca_results: Dict[str, dict], path: str):
    """
    保存 PCA 主成分在 8 个波长通道上的载荷。
    English: save PCA 8 .

    输入:
    English: Input:
        pca_results: 各频层 PCA 结果。
        English: pca_results: PCA result.
        path: 输出 CSV 路径。
        English: path: Output CSV path.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    rows = []
    for band, result in pca_results.items():
        comps = result["component_loadings"]  # [K, 8]
        for i in range(comps.shape[0]):
            row = {
                "band": band,
                "component_index_1based": i + 1
            }
            for j, wl in enumerate(WAVELENGTHS_NUM):
                row[f"wl_{int(wl)}nm"] = float(comps[i, j])
            rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

def save_band_normalization_csv(band_norm_priors: Dict[str, dict], path: str):
    """
    保存各频层固定通道归一化参数。
    English: saveparameter.

    输入:
    English: Input:
        band_norm_priors: 第一遍扫描得到的 norm_center/norm_scale。
        English: band_norm_priors: norm_center/norm_scale.
        path: 输出 CSV 路径。
        English: path: Output CSV path.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    rows = []
    for band, info in band_norm_priors.items():
        row = {
            "band": band,
            "norm_method": info["norm_method"],
            "norm_eps": float(info["norm_eps"]),
            "vector_count": int(info["vector_count"])
        }
        center = np.asarray(info["norm_center"], dtype=np.float32)
        scale = np.asarray(info["norm_scale"], dtype=np.float32)
        for j, wl in enumerate(WAVELENGTHS_NUM):
            row[f"center_{int(wl)}nm"] = float(center[j])
            row[f"scale_{int(wl)}nm"] = float(scale[j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_cross_band_csv(cross_band_analysis: dict, path_prefix: str):
    """
    将跨频层 PCA 差异分析拆分保存为多张 CSV。
    English: PCA save CSV.

    输入:
    English: Input:
        cross_band_analysis: build_cross_band_analysis() 返回的字典。
        English: cross_band_analysis: build_cross_band_analysis() returndictionary.
        path_prefix: 输出文件名前缀。
        English: path_prefix: Outputfile.

    输出:
        *_mean_distance.csv、*_projection_distance.csv、*_principal_angles.csv、
        *_component_cosine.csv。

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """
    # mean distance
    rows = []
    for k, v in cross_band_analysis["mean_l2_distance"].items():
        a, b = k.split("__vs__")
        rows.append({"band_a": a, "band_b": b, "mean_l2_distance": v})
    pd.DataFrame(rows).to_csv(f"{path_prefix}_mean_distance.csv", index=False, encoding="utf-8-sig")

    # projection distance
    rows = []
    for k, v in cross_band_analysis["projection_frobenius_distance"].items():
        a, b = k.split("__vs__")
        rows.append({"band_a": a, "band_b": b, "projection_frobenius_distance": v})
    pd.DataFrame(rows).to_csv(f"{path_prefix}_projection_distance.csv", index=False, encoding="utf-8-sig")

    # angles
    rows = []
    for k, angles in cross_band_analysis["subspace_principal_angles_deg"].items():
        a, b = k.split("__vs__")
        row = {"band_a": a, "band_b": b}
        for i, ang in enumerate(angles):
            row[f"angle_{i+1}_deg"] = float(ang)
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{path_prefix}_principal_angles.csv", index=False, encoding="utf-8-sig")

    # cosine similarity
    rows = []
    for k, mat in cross_band_analysis["component_cosine_similarity_topk"].items():
        a, b = k.split("__vs__")
        mat = np.array(mat, dtype=np.float32)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                rows.append({
                    "band_a": a,
                    "band_b": b,
                    "component_a": i + 1,
                    "component_b": j + 1,
                    "cosine_similarity": float(mat[i, j])
                })
    pd.DataFrame(rows).to_csv(f"{path_prefix}_component_cosine.csv", index=False, encoding="utf-8-sig")


# ================= 9. 主流程：离线 PCA + 统计缓存 + 论文作图数据导出 =================
# EN: ================= 9. program: PCA + cache + image dataexport =================.
# 逻辑：
# EN: Logic:
# 1. 扫描样本并做同级 blank 校正；
# EN: scansample and do same blank;
# 2. 构造拉普拉斯金字塔；
# EN: Laplacian pyramid;
# 3. 高频层做结构向量筛选，低频层直接收集；
# EN: high-frequency levels do structural-vector selection, low-frequency leveldirectly;
# 4. 保存样本级统计与部分代表样本数值缓存；
# EN: savesample and table sample count value cache;
# 5. 对 4 个频层拟合 PCA；
# EN: for 4 frequency bandfit PCA;
# 6. 保存模型所需参数与论文分析所需数值。
# EN: save need parameters and need value.
def build_pca_priors_full(
    data_root,
    output_dir,
    max_samples=None,
    eta_high=DEFAULT_ETA_HIGH,
    eta_low=DEFAULT_ETA_LOW,
    max_components=DEFAULT_MAX_COMPONENTS,
    snr_db=DEFAULT_SNR_DB,
    chi2_quantile=DEFAULT_CHI2_QUANTILE,
    fdr_q=DEFAULT_FDR_Q,
    selection_min_keep_ratio=DEFAULT_SELECTION_MIN_KEEP_RATIO,
    selection_min_keep_channel_multiplier=DEFAULT_SELECTION_MIN_KEEP_CHANNEL_MULTIPLIER,
    selection_min_keep_abs=DEFAULT_SELECTION_MIN_KEEP_ABS,
    background_min_count=DEFAULT_BACKGROUND_MIN_COUNT,
    background_min_channel_multiplier=DEFAULT_BACKGROUND_MIN_CHANNEL_MULTIPLIER,
    background_fallback_base=DEFAULT_BACKGROUND_FALLBACK_BASE,
    background_fallback_channel_multiplier=DEFAULT_BACKGROUND_FALLBACK_CHANNEL_MULTIPLIER,
    max_background_for_cov=DEFAULT_MAX_BACKGROUND_FOR_COV,
    cov_shrinkage_alpha=DEFAULT_COV_SHRINKAGE_ALPHA,
    cov_shrinkage_eps=DEFAULT_COV_SHRINKAGE_EPS,
    compare_topk=3,
    max_visual_samples=32,
    seed=42,
    analysis_target='all',
    soc_mat_path=None,
    tn_mat_path=None,
    crop_high1=DEFAULT_CROP_HIGH1,
    crop_high2=DEFAULT_CROP_HIGH2,
    crop_high3=DEFAULT_CROP_HIGH3,
    crop_low=DEFAULT_CROP_LOW,
    norm_eps=DEFAULT_NORM_EPS
):
    """
    离线构建 MFPC-HFNet 使用的固定通道归一化与 PCA 先验。
    English: build MFPC-HFNet PCA .

    输入:
    English: Input:
        data_root: 原始样本库根目录。
        English: data_root: sampledirectory.
        output_dir: 所有 pt/json/csv/npz 输出目录。
        English: output_dir: pt/json/csv/npz Outputdirectory.
        max_samples: 最多参与构建的样本数；None 表示全部有效样本。
        English: max_samples: buildsample; None sample.
        eta_high / eta_low / max_components: PCA 主成分保留规则。
        English: eta_low / max_components: PCA 主成分保留规则.
        snr_db / chi2_quantile / fdr_q / selection_min_*: 高频结构向量筛选主控参数。
        English: chi2_quantile / fdr_q / selection_min_*: 高频结构向量筛选主控参数.
        background_* / cov_shrinkage_*: 背景分布建模和协方差稳定参数。
        English: cov_shrinkage_*: 背景分布建模和协方差稳定参数.
        compare_topk / max_visual_samples: 论文分析数值缓存控制项。
        English: max_visual_samples: 论文分析数值缓存控制项.
        analysis_target / soc_mat_path / tn_mat_path: 按真值可用性限定样本范围。
        English: soc_mat_path / tn_mat_path: 按真值可用性限定样本范围.
        crop_high1/crop_high2/crop_high3/crop_low: 金字塔后中心裁切尺寸。
        English: crop_high1/crop_high2/crop_high3/crop_low: .
        norm_eps: 固定通道归一化的数值稳定项。
        English: norm_eps: .

    输出:
    English: Output:
        不返回值；在 output_dir 下保存:
            pca_priors_full.pt、summary.json、stats/*.csv、tables/*.csv、visual_cache/*.npz。

    复现注意:
    English: :
        本函数执行两遍样本遍历。第一遍统计 norm_center/norm_scale，第二遍在固定归一化空间中筛选结构向量并累计 PCA。
        English: sample. norm_center/norm_scale, PCA.
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
    """
    set_seed(seed)

    start = time.time()

    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "stats"))
    ensure_dir(os.path.join(output_dir, "visual_cache"))
    ensure_dir(os.path.join(output_dir, "tables"))

    scanner = RawImageSampleScanner(data_root)

    soc_dict, tn_dict = load_ground_truth_dicts(
      soc_mat_path=soc_mat_path,
    tn_mat_path=tn_mat_path
    )

    records, scan_report, duplicate_items = scanner.scan(
        analysis_target=analysis_target,
        soc_dict=soc_dict,
        tn_dict=tn_dict
    )

    print("=" * 70)
    print(">> 离线 PCA + 统计缓存 + 论文作图数据导出")
    print(f"   数据根目录: {data_root}")
    print(f"   扫描批次数: {scan_report['scanned_groups']}")
    print(f"   有效样本记录数: {scan_report['valid_records']}")
    print(f"   重复完整样本名数: {scan_report['duplicate_sample_names']}")
    print("=" * 70)

    if len(records) == 0:
        raise RuntimeError("未找到有效样本，无法继续。")

    if max_samples is not None:
        effective_total = min(len(records), max_samples)
    else:
        effective_total = len(records)

    visual_cache_indices = choose_visual_cache_indices(
        total_count=effective_total,
        max_visual_samples=max_visual_samples,
        seed=seed
    )

    crop_size_map = {
        "high1": crop_high1,
        "high2": crop_high2,
        "high3": crop_high3,
        "low": crop_low
    }

    print(">> [Pass-1/2] 统计各频层逐通道归一化参数")
    band_norm_priors = build_band_normalization_priors(
        scanner=scanner,
        records=records,
        max_samples=max_samples,
        crop_size_map=crop_size_map,
        norm_eps=norm_eps
    )
    print(">> [Pass-1/2] 归一化统计完成")
    for band_name in ["high1", "high2", "high3", "low"]:
        center = band_norm_priors[band_name]["norm_center"]
        scale = band_norm_priors[band_name]["norm_scale"]
        print(
            f"   {band_name}: center_mean={float(center.mean()):.4f}, "
            f"scale_mean={float(scale.mean()):.4f}, vectors={band_norm_priors[band_name]['vector_count']}"
        )

    band_stats = {
        "high1": RunningPCAStats(channel_dim=len(WAVELENGTHS)),
        "high2": RunningPCAStats(channel_dim=len(WAVELENGTHS)),
        "high3": RunningPCAStats(channel_dim=len(WAVELENGTHS)),
        "low": RunningPCAStats(channel_dim=len(WAVELENGTHS))
    }

    sample_level_rows = []
    visual_cache_items = []
    processed = 0

    print(">> [Pass-2/2] 基于固定归一化先验执行结构筛选与 PCA 统计")
    for idx, item in enumerate(scanner.iter_calibrated_images(records, max_samples=max_samples)):
        img = item["image"]  # [8, H, W]

        try:
            pyramid = build_laplacian_pyramid(img, num_levels=3)

            high1_feat = normalize_feature_map_channels(
                prepare_band_feature_map(pyramid, "high1", crop_size_map),
                band_norm_priors["high1"]["norm_center"],
                band_norm_priors["high1"]["norm_scale"],
                eps=band_norm_priors["high1"]["norm_eps"]
            )
            high2_feat = normalize_feature_map_channels(
                prepare_band_feature_map(pyramid, "high2", crop_size_map),
                band_norm_priors["high2"]["norm_center"],
                band_norm_priors["high2"]["norm_scale"],
                eps=band_norm_priors["high2"]["norm_eps"]
            )
            high3_feat = normalize_feature_map_channels(
                prepare_band_feature_map(pyramid, "high3", crop_size_map),
                band_norm_priors["high3"]["norm_center"],
                band_norm_priors["high3"]["norm_scale"],
                eps=band_norm_priors["high3"]["norm_eps"]
            )
            low_feat = normalize_feature_map_channels(
                prepare_band_feature_map(pyramid, "low", crop_size_map),
                band_norm_priors["low"]["norm_center"],
                band_norm_priors["low"]["norm_scale"],
                eps=band_norm_priors["low"]["norm_eps"]
            )

            # 高频筛选在归一化空间执行，避免能量统计被大方差通道主导。
            # EN: high in execute, avoid can amount large pass.
            high1_vec, high1_info, _, aux1 = select_structural_vectors(
                high1_feat,
                snr_db=snr_db,
                chi2_quantile=chi2_quantile,
                fdr_q=fdr_q,
                min_keep_ratio=selection_min_keep_ratio,
                min_keep_channel_multiplier=selection_min_keep_channel_multiplier,
                min_keep_abs=selection_min_keep_abs,
                background_min_count=background_min_count,
                background_min_channel_multiplier=background_min_channel_multiplier,
                background_fallback_base=background_fallback_base,
                background_fallback_channel_multiplier=background_fallback_channel_multiplier,
                max_background_for_cov=max_background_for_cov,
                cov_shrinkage_alpha=cov_shrinkage_alpha,
                cov_shrinkage_eps=cov_shrinkage_eps
            )
            high2_vec, high2_info, _, aux2 = select_structural_vectors(
                high2_feat,
                snr_db=snr_db,
                chi2_quantile=chi2_quantile,
                fdr_q=fdr_q,
                min_keep_ratio=selection_min_keep_ratio,
                min_keep_channel_multiplier=selection_min_keep_channel_multiplier,
                min_keep_abs=selection_min_keep_abs,
                background_min_count=background_min_count,
                background_min_channel_multiplier=background_min_channel_multiplier,
                background_fallback_base=background_fallback_base,
                background_fallback_channel_multiplier=background_fallback_channel_multiplier,
                max_background_for_cov=max_background_for_cov,
                cov_shrinkage_alpha=cov_shrinkage_alpha,
                cov_shrinkage_eps=cov_shrinkage_eps
            )
            high3_vec, high3_info, _, aux3 = select_structural_vectors(
                high3_feat,
                snr_db=snr_db,
                chi2_quantile=chi2_quantile,
                fdr_q=fdr_q,
                min_keep_ratio=selection_min_keep_ratio,
                min_keep_channel_multiplier=selection_min_keep_channel_multiplier,
                min_keep_abs=selection_min_keep_abs,
                background_min_count=background_min_count,
                background_min_channel_multiplier=background_min_channel_multiplier,
                background_fallback_base=background_fallback_base,
                background_fallback_channel_multiplier=background_fallback_channel_multiplier,
                max_background_for_cov=max_background_for_cov,
                cov_shrinkage_alpha=cov_shrinkage_alpha,
                cov_shrinkage_eps=cov_shrinkage_eps
            )

            # 低频层不做结构筛选，但同样先进入归一化空间。
            # EN: low-frequency level not do result, same first.
            low_vec = vectorize_feature_map(low_feat)

            band_stats["high1"].update(high1_vec)
            band_stats["high2"].update(high2_vec)
            band_stats["high3"].update(high3_vec)
            band_stats["low"].update(low_vec)


            soc_value = item.get("soc_value", None)
            tn_value = item.get("tn_value", None)

            row = {
                "sample_name": item["sample_name"],
                "core_id": item["core_id"],
                "folder_path": item["folder_path"],
                "blank_path": item["blank_path"],
                "orig_h": int(img.shape[1]),
                "orig_w": int(img.shape[2]),
                "crop_high1": np.nan if crop_high1 is None else int(crop_high1),
                "crop_high2": np.nan if crop_high2 is None else int(crop_high2),
                "crop_high3": np.nan if crop_high3 is None else int(crop_high3),
                "crop_low": np.nan if crop_low is None else int(crop_low),
                "channel_norm_method": "zscore_mean_std",

                "has_soc": int(item.get("has_soc", False)),
                "has_tn": int(item.get("has_tn", False)),
                "soc_value": float(soc_value) if soc_value is not None else np.nan,
                "tn_value": float(tn_value) if tn_value is not None else np.nan,

                "high1_total_vectors": high1_info["total_vectors"],
                "high1_kept_count": high1_info["kept_count"],
                "high1_kept_ratio": high1_info["kept_ratio"],
                "high1_energy_q50": high1_info["energy_q50"],
                "high1_energy_q90": high1_info["energy_q90"],
                "high1_energy_q99": high1_info["energy_q99"],
                "high1_d2_q50": high1_info["d2_q50"],
                "high1_d2_q90": high1_info["d2_q90"],
                "high1_d2_q99": high1_info["d2_q99"],

                "high2_total_vectors": high2_info["total_vectors"],
                "high2_kept_count": high2_info["kept_count"],
                "high2_kept_ratio": high2_info["kept_ratio"],
                "high2_energy_q50": high2_info["energy_q50"],
                "high2_energy_q90": high2_info["energy_q90"],
                "high2_energy_q99": high2_info["energy_q99"],
                "high2_d2_q50": high2_info["d2_q50"],
                "high2_d2_q90": high2_info["d2_q90"],
                "high2_d2_q99": high2_info["d2_q99"],

                "high3_total_vectors": high3_info["total_vectors"],
                "high3_kept_count": high3_info["kept_count"],
                "high3_kept_ratio": high3_info["kept_ratio"],
                "high3_energy_q50": high3_info["energy_q50"],
                "high3_energy_q90": high3_info["energy_q90"],
                "high3_energy_q99": high3_info["energy_q99"],
                "high3_d2_q50": high3_info["d2_q50"],
                "high3_d2_q90": high3_info["d2_q90"],
                "high3_d2_q99": high3_info["d2_q99"],

                "low_total_vectors": int(low_vec.shape[0]),
                "low_kept_count": int(low_vec.shape[0]),
                "low_kept_ratio": 1.0
            }
            
            sample_level_rows.append(row)

            if idx in visual_cache_indices:
                visual_cache_items.append({
                    "item": item,
                    "pyramid": pyramid,
                    "normalized_band_features": {
                        "high1": high1_feat,
                        "high2": high2_feat,
                        "high3": high3_feat,
                        "low": low_feat
                    },
                    "aux_maps": {
                        "high1": aux1,
                        "high2": aux2,
                        "high3": aux3
                    }
                })

            processed += 1
            if processed % 20 == 0:
                print(f">> [Pass-2/2] 已处理样本: {processed}")

        except Exception as e:
            print(f"[Warning] 样本处理失败: {item['folder_path']} | {e}")

    if processed == 0:
        raise RuntimeError("没有成功处理任何样本。")

    print("=" * 70)
    print(f">> 样本处理完成，共成功处理 {processed} 个样本")
    print("=" * 70)

    for band_name, stats in band_stats.items():
        print(f">> {band_name}: merged vectors count = {stats.count}, channel_dim = {stats.channel_dim}")

    # ---------- PCA 拟合 ----------
    # EN: ---------- PCA fit ----------.
    pca_results = {}
    for band_name in ["high1", "high2", "high3"]:
        pca_results[band_name] = fit_pca_from_running_stats(
            band_stats[band_name],
            eta=eta_high,
            max_components=max_components
        )

    pca_results["low"] = fit_pca_from_running_stats(
        band_stats["low"],
        eta=eta_low,
        max_components=max_components
    )

    # ---------- 频层汇总统计 ----------
    # EN: ---------- frequency band ----------.
    band_summary_rows = []
    for band_name in ["high1", "high2", "high3", "low"]:
        result = pca_results[band_name]
        stats = band_stats[band_name]

        row = {
            "band": band_name,
            "merged_vector_count": int(stats.count),
            "channel_dim": int(stats.channel_dim),
            "selected_k": int(result["selected_k"]),
            "eigval_1": float(result["eigvals_full"][0]),
            "explained_ratio_1": float(result["explained_variance_ratio_full"][0]),
            "cumulative_ratio_k": float(result["cumulative_explained_variance"][-1]),
            "eta_target": float(eta_high if band_name != "low" else eta_low),
            "norm_method": band_norm_priors[band_name]["norm_method"],
            "norm_center_mean": float(np.mean(band_norm_priors[band_name]["norm_center"])),
            "norm_scale_mean": float(np.mean(band_norm_priors[band_name]["norm_scale"]))
        }

        # 记录达到若干阈值时所需主成分数
        # EN: to if threshold when need principal components number.
        cumsum = result["cumulative_explained_variance_full"]
        for thr in [0.80, 0.90, 0.95, 0.99]:
            k_thr = int(np.searchsorted(cumsum, thr) + 1)
            row[f"k_at_{str(thr).replace('.', '_')}"] = k_thr

        band_summary_rows.append(row)

    # ---------- PCASE 所需有效特征向量占比统计 ----------
    # EN: ---------- PCASE need amount ----------.
    # 逻辑：
    # EN: Logic:
    # 1. high1/high2/high3 的 N_l 来自离线结构向量筛选结果；
    # EN: high1/high2/high3 N_l structural-vector selection result result;
    # 2. 这里按所有成功样本的 kept_count / total_vectors 计算全局有效特征向量占比；
    # EN: here by sample kept_count / total_vectors full amount;
    # 3. low 层不执行结构向量筛选，因此 N_l 固定为 1.0；
    # EN: low not executestructural-vector selection, therefore N_l fixed as 1.0;
    # 4. 这些字段会写入 pca_priors_full.pt，供 MFPC-HFNet 训练时直接读取，
    # EN: field will write pca_priors_full.pt, MFPC-HFNet train when directlyread,.
    #    避免继续退回固定 input_feature_vector_ratio。
    # EN: avoid fixed input_feature_vector_ratio.
    pcase_vector_stats = {}
    sample_stats_df_for_ratio = pd.DataFrame(sample_level_rows)
    for band_name in ["high1", "high2", "high3"]:
        total_col = f"{band_name}_total_vectors"
        kept_col = f"{band_name}_kept_count"
        total_candidate_vectors = int(sample_stats_df_for_ratio[total_col].sum())
        effective_vector_count = int(sample_stats_df_for_ratio[kept_col].sum())
        effective_feature_vector_ratio = (
            float(effective_vector_count) / float(total_candidate_vectors)
            if total_candidate_vectors > 0 else 0.0
        )
        pcase_vector_stats[band_name] = {
            "total_candidate_vectors": total_candidate_vectors,
            "effective_vector_count": effective_vector_count,
            "effective_feature_vector_ratio": float(effective_feature_vector_ratio),
            "feature_vector_ratio_source": "offline_structural_vector_screening",
        }

    low_total_vectors = int(sample_stats_df_for_ratio["low_total_vectors"].sum())
    pcase_vector_stats["low"] = {
        "total_candidate_vectors": low_total_vectors,
        "effective_vector_count": low_total_vectors,
        "effective_feature_vector_ratio": 1.0,
        "feature_vector_ratio_source": "fixed_low_no_vector_screening",
    }

    for row in band_summary_rows:
        band_name = row["band"]
        row.update(pcase_vector_stats[band_name])

    # ---------- 跨频层差异分析 ----------
    # EN: ---------- frequency band ----------.
    cross_band_analysis = build_cross_band_analysis(
        pca_results=pca_results,
        compare_topk=compare_topk
    )

    # ---------- 代表样本 PCA 映射缓存 ----------
    # EN: ---------- table sample PCA cache ----------.
    # 为避免后续重跑，这里把代表样本在各频层的前3个 PCA 投影图也缓存为数值
    # EN: as avoidlater, here table sample in each frequency band before 3 PCA image also cache as value.
    for cache_item in visual_cache_items:
        pca_maps = {}
        for band_name in ["high1", "high2", "high3", "low"]:
            feat = cache_item["normalized_band_features"][band_name]
            result = pca_results[band_name]
            pca_maps[band_name] = project_feature_map_with_pca(
                feat_chw=feat,
                mean=result["mean"],
                components=result["components"],
                topk=3
            )

        save_numeric_visual_cache(
            save_dir=os.path.join(output_dir, "visual_cache"),
            item=cache_item["item"],
            pyramid=cache_item["pyramid"],
            aux_maps=cache_item["aux_maps"],
            pca_maps=pca_maps
        )

    # ---------- 构造总 priors 包 ----------
    # EN: ---------- priors ----------.
    priors = {
        "meta": {
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_root": data_root,
            "processed_samples": int(processed),
            "scan_report": scan_report,
            "snr_db": float(snr_db),
            "chi2_quantile": float(chi2_quantile),
            "fdr_q": float(fdr_q),
            "selection_min_keep_ratio": float(selection_min_keep_ratio),
            "selection_min_keep_channel_multiplier": int(selection_min_keep_channel_multiplier),
            "selection_min_keep_abs": int(selection_min_keep_abs),
            "background_min_count": int(background_min_count),
            "background_min_channel_multiplier": int(background_min_channel_multiplier),
            "background_fallback_base": int(background_fallback_base),
            "background_fallback_channel_multiplier": int(background_fallback_channel_multiplier),
            "max_background_for_cov": int(max_background_for_cov),
            "cov_shrinkage_alpha": float(cov_shrinkage_alpha),
            "cov_shrinkage_eps": float(cov_shrinkage_eps),
            "eta_high": float(eta_high),
            "eta_low": float(eta_low),
            "max_components": None if max_components is None else int(max_components),
            "compare_topk": int(compare_topk),
            "max_visual_samples": int(max_visual_samples),
            "selection_parameter_priority": {
                "P1_direct_kept_count": [
                    "crop_high1", "crop_high2", "crop_high3", "crop_low",
                    "snr_db", "chi2_quantile", "fdr_q",
                    "selection_min_keep_ratio", "selection_min_keep_channel_multiplier", "selection_min_keep_abs"
                ],
                "P2_background_model_stability": [
                    "background_min_count", "background_min_channel_multiplier",
                    "background_fallback_base", "background_fallback_channel_multiplier",
                    "max_background_for_cov", "cov_shrinkage_alpha", "cov_shrinkage_eps"
                ],
                "P3_direct_selected_k": [
                    "eta_high", "eta_low", "max_components"
                ],
                "notes": [
                    "P1 先决定 kept_count，再通过进入 PCA 的向量集合间接影响 selected_k。",
                    "P2 主要影响背景协方差估计稳定性，属于 kept_count 的二级调节项。",
                    "norm_eps 主要用于数值稳定，通常不作为调数量参数。"
                ]
            },
            "crop_high1": None if crop_high1 is None else int(crop_high1),
            "crop_high2": None if crop_high2 is None else int(crop_high2),
            "crop_high3": None if crop_high3 is None else int(crop_high3),
            "crop_low": None if crop_low is None else int(crop_low),
            "wavelengths_nm": WAVELENGTHS_NUM.tolist(),
            "channel_norm_method": "zscore_mean_std",
            "channel_norm_eps": float(norm_eps),
            "pcase_vector_stats": pcase_vector_stats,
            "note": "仅保存数值缓存；先对整图构建拉普拉斯金字塔，再对各频层执行中心裁切与固定通道归一化；high1/high2/high3 在归一化空间中进行结构向量筛选并保存 effective_feature_vector_ratio；low 频层在归一化空间中直接做 PCA，effective_feature_vector_ratio 固定为 1.0。",
            "analysis_target": str(analysis_target),
            "soc_mat_path": soc_mat_path,
            "tn_mat_path": tn_mat_path,
        },
        "cross_band_analysis": cross_band_analysis
    }

    for band_name in ["high1", "high2", "high3", "low"]:
        result = pca_results[band_name]
        priors[band_name] = {
            "mean": torch.from_numpy(result["mean"]),
            "components": torch.from_numpy(result["components"]),
            "eigvals": torch.from_numpy(result["eigvals"]),
            "explained_variance_ratio": torch.from_numpy(result["explained_variance_ratio"]),
            "cumulative_explained_variance": torch.from_numpy(result["cumulative_explained_variance"]),
            "eigvals_full": torch.from_numpy(result["eigvals_full"]),
            "explained_variance_ratio_full": torch.from_numpy(result["explained_variance_ratio_full"]),
            "cumulative_explained_variance_full": torch.from_numpy(result["cumulative_explained_variance_full"]),
            "weight": torch.from_numpy(result["weight"]),
            "bias": torch.from_numpy(result["bias"]),
            "covariance": torch.from_numpy(result["covariance"]),
            "component_loadings": torch.from_numpy(result["component_loadings"]),
            "norm_method": band_norm_priors[band_name]["norm_method"],
            "norm_center": torch.from_numpy(band_norm_priors[band_name]["norm_center"]),
            "norm_scale": torch.from_numpy(band_norm_priors[band_name]["norm_scale"]),
            "norm_eps": float(band_norm_priors[band_name]["norm_eps"]),
            "selected_k": int(result["selected_k"]),
            "full_dim": int(result["full_dim"]),
            "total_candidate_vectors": int(pcase_vector_stats[band_name]["total_candidate_vectors"]),
            "effective_vector_count": int(pcase_vector_stats[band_name]["effective_vector_count"]),
            "effective_feature_vector_ratio": float(pcase_vector_stats[band_name]["effective_feature_vector_ratio"]),
            "feature_vector_ratio_source": pcase_vector_stats[band_name]["feature_vector_ratio_source"]
        }

    priors_path = os.path.join(output_dir, "pca_priors_full.pt")
    torch.save(priors, priors_path)

    # ---------- 保存各种分析表 ----------
    # EN: ---------- save each table ----------.
    sample_stats_csv = os.path.join(output_dir, "tables", "sample_level_stats.csv")
    band_summary_csv = os.path.join(output_dir, "tables", "band_level_summary.csv")
    scree_csv = os.path.join(output_dir, "tables", "pca_scree_full.csv")
    component_loadings_csv = os.path.join(output_dir, "tables", "component_loadings.csv")
    band_normalization_csv = os.path.join(output_dir, "tables", "band_channel_normalization.csv")
    cross_prefix = os.path.join(output_dir, "tables", "cross_band")

    save_sample_stats_csv(sample_level_rows, sample_stats_csv)
    save_band_summary_csv(band_summary_rows, band_summary_csv)
    save_scree_csv(pca_results, scree_csv)
    save_component_loadings_csv(pca_results, component_loadings_csv)
    save_band_normalization_csv(band_norm_priors, band_normalization_csv)
    save_cross_band_csv(cross_band_analysis, cross_prefix)

    # ---------- 保存结构化摘要 ----------
    # EN: ---------- save result need ----------.
    summary = {
        "meta": priors["meta"],
        "bands": {},
        "output_files": {
            "priors_pt": "pca_priors_full.pt",
            "sample_level_stats_csv": "tables/sample_level_stats.csv",
            "band_level_summary_csv": "tables/band_level_summary.csv",
            "pca_scree_full_csv": "tables/pca_scree_full.csv",
            "component_loadings_csv": "tables/component_loadings.csv",
            "band_channel_normalization_csv": "tables/band_channel_normalization.csv",
            "cross_band_mean_distance_csv": "tables/cross_band_mean_distance.csv",
            "cross_band_projection_distance_csv": "tables/cross_band_projection_distance.csv",
            "cross_band_principal_angles_csv": "tables/cross_band_principal_angles.csv",
            "cross_band_component_cosine_csv": "tables/cross_band_component_cosine.csv",
            "visual_cache_dir": "visual_cache/"
        }
    }

    for band_name in ["high1", "high2", "high3", "low"]:
        summary["bands"][band_name] = {
            "selected_k": int(pca_results[band_name]["selected_k"]),
            "full_dim": int(pca_results[band_name]["full_dim"]),
            "merged_vector_count": int(band_stats[band_name].count),
            "components_shape": list(pca_results[band_name]["components"].shape),
            "weight_shape": list(pca_results[band_name]["weight"].shape),
            "bias_shape": list(pca_results[band_name]["bias"].shape),
            "norm_center_shape": list(np.asarray(band_norm_priors[band_name]["norm_center"]).shape),
            "norm_scale_shape": list(np.asarray(band_norm_priors[band_name]["norm_scale"]).shape),
            "total_candidate_vectors": int(pcase_vector_stats[band_name]["total_candidate_vectors"]),
            "effective_vector_count": int(pcase_vector_stats[band_name]["effective_vector_count"]),
            "effective_feature_vector_ratio": float(pcase_vector_stats[band_name]["effective_feature_vector_ratio"]),
            "feature_vector_ratio_source": pcase_vector_stats[band_name]["feature_vector_ratio_source"]
        }

    save_json(summary, os.path.join(output_dir, "summary.json"))
    save_json(scan_report, os.path.join(output_dir, "stats", "scan_report.json"))

    duplicate_df = pd.DataFrame(duplicate_items)
    duplicate_df.to_csv(
        os.path.join(output_dir, "stats", "duplicate_sample_items.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 70)
    print(">> 全部完成")
    print(f"   priors:        {priors_path}")
    print(f"   summary:       {os.path.join(output_dir, 'summary.json')}")
    print(f"   visual_cache:  {os.path.join(output_dir, 'visual_cache')}")
    print(f"   elapsed:       {time.time() - start:.1f}s")
    print("=" * 70)


# ================= 10. 命令行入口 =================
# EN: ================= 10. line interface =================.
# 逻辑：
# EN: Logic:
# 1. 保持单脚本独立可运行；
# EN: single can run;
# 2. 所有关键统计参数都可在命令行覆盖；
# EN: parameters all can in line override;
# 3. 默认行为偏保守，优先保证论文分析可复现。
# EN: default line as, preferensure can.
def parse_args():
    """
    构建命令行参数解析器并返回解析结果。
    English: buildparameterparsereturnparseresult.

    输出:
    English: Output:
        argparse.Namespace，字段会直接传入 build_pca_priors_full()。
        English: argparse.Namespace, field build_pca_priors_full().

    说明:
    English: :
        直接运行脚本时，默认参数来自文件顶部配置区；命令行覆盖只影响本次运行，不回写源码参数。
        English: , defaultparameterfileconfiguration; , parameter.
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
    """
    parser = argparse.ArgumentParser(description="离线 PCA + 统计缓存 + 论文作图数据导出")

    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="原始样本库根目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录"
    )

    parser.add_argument(
        "--analysis_target",
        type=str,
        default=DEFAULT_ANALYSIS_TARGET,
        choices=["all", "soc", "tn", "both"],
        help="按真值可用性筛选分析样本"
    )
    parser.add_argument(
        "--soc_mat_path",
        type=str,
        default=DEFAULT_SOC_MAT_PATH,
        help="SOC.mat 路径"
    )
    parser.add_argument(
        "--tn_mat_path",
        type=str,
        default=DEFAULT_TN_MAT_PATH,
        help="TN.mat 路径"
    )

    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES, help="最多处理多少样本，默认全部")
    parser.add_argument("--crop_high1", type=int, default=DEFAULT_CROP_HIGH1, help="high1 金字塔后中心裁切尺寸，默认 800")
    parser.add_argument("--crop_high2", type=int, default=DEFAULT_CROP_HIGH2, help="high2 金字塔后中心裁切尺寸，默认 400")
    parser.add_argument("--crop_high3", type=int, default=DEFAULT_CROP_HIGH3, help="high3 金字塔后中心裁切尺寸，默认 200")
    parser.add_argument("--crop_low", type=int, default=-1, help="low 频层金字塔后中心裁切尺寸；默认 -1 表示不裁切")
    parser.add_argument("--eta_high", type=float, default=DEFAULT_ETA_HIGH, help="高频层累计解释方差阈值；直接影响 high1/high2/high3 的 selected_k")
    parser.add_argument("--eta_low", type=float, default=DEFAULT_ETA_LOW, help="低频层累计解释方差阈值；直接影响 low 的 selected_k")
    parser.add_argument("--max_components", type=int, default=DEFAULT_MAX_COMPONENTS, help="PCA 最大保留维数；对 selected_k 施加硬上限")

    parser.add_argument("--snr_db", type=float, default=DEFAULT_SNR_DB, help="能量统计中的噪声强度假设；P1，直接影响结构向量初筛")
    parser.add_argument("--chi2_quantile", type=float, default=DEFAULT_CHI2_QUANTILE, help="卡方能量初筛分位数；P1，直接影响 background/structure 候选划分")
    parser.add_argument("--fdr_q", type=float, default=DEFAULT_FDR_Q, help="BH-FDR 容忍度；P1，直接影响正式保留的结构向量数量")
    parser.add_argument("--selection_min_keep_ratio", type=float, default=DEFAULT_SELECTION_MIN_KEEP_RATIO, help="结构向量保底比例；P1，当 FDR 保留过少时直接接管 kept_count")
    parser.add_argument("--selection_min_keep_channel_multiplier", type=int, default=DEFAULT_SELECTION_MIN_KEEP_CHANNEL_MULTIPLIER, help="结构向量保底的通道倍率；P1")
    parser.add_argument("--selection_min_keep_abs", type=int, default=DEFAULT_SELECTION_MIN_KEEP_ABS, help="结构向量保底的绝对下限；P1")
    parser.add_argument("--background_min_count", type=int, default=DEFAULT_BACKGROUND_MIN_COUNT, help="背景候选最小数量；P2")
    parser.add_argument("--background_min_channel_multiplier", type=int, default=DEFAULT_BACKGROUND_MIN_CHANNEL_MULTIPLIER, help="背景候选最小数量的通道倍率；P2")
    parser.add_argument("--background_fallback_base", type=int, default=DEFAULT_BACKGROUND_FALLBACK_BASE, help="背景候选不足时，按能量最低向量回填的基础数量；P2")
    parser.add_argument("--background_fallback_channel_multiplier", type=int, default=DEFAULT_BACKGROUND_FALLBACK_CHANNEL_MULTIPLIER, help="背景候选不足时，按通道倍率回填的数量；P2")
    parser.add_argument("--max_background_for_cov", type=int, default=DEFAULT_MAX_BACKGROUND_FOR_COV, help="用于背景协方差估计的最大背景向量数；P2")
    parser.add_argument("--cov_shrinkage_alpha", type=float, default=DEFAULT_COV_SHRINKAGE_ALPHA, help="背景协方差收缩强度；P2")
    parser.add_argument("--cov_shrinkage_eps", type=float, default=DEFAULT_COV_SHRINKAGE_EPS, help="背景协方差数值稳定项；P2")
    parser.add_argument("--norm_eps", type=float, default=DEFAULT_NORM_EPS, help="固定通道归一化时的最小数值稳定项；通常不用于调数量")

    parser.add_argument("--compare_topk", type=int, default=DEFAULT_COMPARE_TOPK, help="跨频层子空间比较时取前几个主成分")
    parser.add_argument("--max_visual_samples", type=int, default=DEFAULT_MAX_VISUAL_SAMPLES, help="最多缓存多少个代表样本的完整数值")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    try:
        build_pca_priors_full(
            data_root=args.data_root,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            eta_high=args.eta_high,
            eta_low=args.eta_low,
            max_components=args.max_components,
            snr_db=args.snr_db,
            chi2_quantile=args.chi2_quantile,
            fdr_q=args.fdr_q,
            selection_min_keep_ratio=args.selection_min_keep_ratio,
            selection_min_keep_channel_multiplier=args.selection_min_keep_channel_multiplier,
            selection_min_keep_abs=args.selection_min_keep_abs,
            background_min_count=args.background_min_count,
            background_min_channel_multiplier=args.background_min_channel_multiplier,
            background_fallback_base=args.background_fallback_base,
            background_fallback_channel_multiplier=args.background_fallback_channel_multiplier,
            max_background_for_cov=args.max_background_for_cov,
            cov_shrinkage_alpha=args.cov_shrinkage_alpha,
            cov_shrinkage_eps=args.cov_shrinkage_eps,
            compare_topk=args.compare_topk,
            max_visual_samples=args.max_visual_samples,
            seed=args.seed,
            analysis_target=args.analysis_target,
            soc_mat_path=args.soc_mat_path,
            tn_mat_path=args.tn_mat_path,
            crop_high1=None if args.crop_high1 <= 0 else args.crop_high1,
            crop_high2=None if args.crop_high2 <= 0 else args.crop_high2,
            crop_high3=None if args.crop_high3 <= 0 else args.crop_high3,
            crop_low=None if args.crop_low <= 0 else args.crop_low,
            norm_eps=args.norm_eps
        )
    except Exception as e:
        print("\n[ERROR] 脚本执行失败")
        print(str(e))
        traceback.print_exc()
        raise


