# -*- coding: utf-8 -*-
# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
"""
训练 optimizer 策略工具。
Training optimizer policy utilities.

逻辑说明 / Logic
English: Logic.
----------------
1. 本文件属于 Train 层，负责把菜单 `ModelSpec.optimizer_policy` 翻译为可执行 optimizer 行为。
English: 1. file Train , menu `ModelSpec.optimizer_policy` optimizer .
2. 菜单只声明策略意图；模型层只提供参数分组语义；本文件统一执行 AdamW、分组学习率和冻结预热。
English: 2. menu; modelparameter groups; file AdamW, learning rate.
3. 训练引擎调用本文件时不需要知道 SwinV2、ConvNeXt 或 timm/torchvision 的内部参数命名。
English: 3. training enginefile SwinV2, ConvNeXt timm/torchvision parameter.
4. 当前支持 `default_adamw`、`layerwise_lr` 和 `freeze_then_layerwise`，并允许菜单通过 `extra` 扩展任意 lr_role 的学习率。
English: 4. current `default_adamw`, `layerwise_lr` `freeze_then_layerwise`, menu `extra` lr_role learning rate.

最近修改时间 / Last modified: 2026-05-29
English: Last modified: 2026-05-29.
作者 / Author: ljy
English: Author: ljy.
"""

from __future__ import annotations

from typing import Any, Iterable


DEFAULT_OPTIMIZER_POLICY = "default_adamw"
LAYERWISE_POLICIES = {"layerwise_lr", "freeze_then_layerwise"}


def normalize_optimizer_policy(policy: str | None) -> str:
    """
    规范化菜单传入的 optimizer 策略名称。
    English: Normalize the optimizer policy name passed by the menu.

    输入:
    English: Input:
        policy: 菜单 `ModelSpec.optimizer_policy` 字段。
        English: policy: menu `ModelSpec.optimizer_policy` field.
    输出:
    English: Output:
        标准策略名称；空值统一视为 `default_adamw`。
        English: name; empty value `default_adamw`.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    text = str(policy or "").strip().lower()
    if text in {"", "default", "adamw"}:
        return DEFAULT_OPTIMIZER_POLICY
    return text


def _list_params(params: Iterable[Any]) -> list[Any]:
    """
    把参数迭代器转为列表，并过滤空项。
    English: Convert a parameter iterator to a list and filter empty items.
    """

    return [param for param in params if param is not None]


def _deduplicate_params(params: Iterable[Any], seen: set[int]) -> list[Any]:
    """
    按对象 id 去重，避免同一参数进入多个 optimizer group。
    English: Deduplicate by object id so the same parameter does not enter multiple optimizer groups.
    """

    unique = []
    for param in params:
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        unique.append(param)
    return unique


def collect_model_optimizer_groups(model: Any) -> list[dict[str, Any]]:
    """
    从模型对象读取通用 optimizer 参数分组。
    English: Read general optimizer parameter groups from a model object.

    设计说明:
    English: Design note:
    - 若模型实现 `get_optimizer_parameter_groups()`，则使用模型提供的语义分组；
    English: - model `get_optimizer_parameter_groups()`, model;
    - 否则回退到单一 `all` 参数组，保证普通模型仍可复用默认 AdamW。
    English: - fall back `all` parameter, ensuremodeldefault AdamW.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    provider = getattr(model, "get_optimizer_parameter_groups", None)
    raw_groups = provider() if callable(provider) else [{"name": "all", "lr_role": "default", "params": model.parameters()}]
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw_group in enumerate(raw_groups, start=1):
        params = _deduplicate_params(_list_params(raw_group.get("params", [])), seen)
        if not params:
            continue
        name = str(raw_group.get("name") or f"group_{index}")
        role = str(raw_group.get("lr_role") or name).strip().lower()
        groups.append({
            "name": name,
            "lr_role": role,
            "params": params,
            "param_count": int(sum(param.numel() for param in params)),
        })
    if not groups:
        params = _list_params(model.parameters())
        groups.append({
            "name": "all",
            "lr_role": "default",
            "params": params,
            "param_count": int(sum(param.numel() for param in params)),
        })
    return groups


def _get_spec_extra(spec: Any) -> dict[str, Any]:
    """
    读取 ModelSpec.extra，缺失时返回空字典。
    English: Read `ModelSpec.extra`; return an empty dictionary when it is missing.
    """

    extra = getattr(spec, "extra", None)
    return dict(extra) if isinstance(extra, dict) else {}


def resolve_lr_for_role(role: str, spec: Any, config: Any) -> float:
    """
    解析某个参数角色的学习率。
    English: Resolve the learning rate for a parameter role.

    优先级:
    English: Priority:
    1. `ModelSpec.extra["optimizer_role_lrs"][role]` 显式值；
    2. `ModelSpec.extra["optimizer_role_lr_scales"][role] × Config.LEARNING_RATE`；
    3. 兼容字段：`backbone` 用 `backbone_lr`，其他角色用 `head_lr`；
    English: 3. compatiblefield: `backbone` `backbone_lr`, `head_lr`;
    4. 回退到 `Config.LEARNING_RATE`。
    English: 4. fall back `Config.LEARNING_RATE`.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    role_key = str(role or "default").strip().lower()
    extra = _get_spec_extra(spec)
    base_lr = float(getattr(config, "LEARNING_RATE", 1e-4))
    role_lrs = extra.get("optimizer_role_lrs", {})
    if isinstance(role_lrs, dict) and role_key in role_lrs:
        return float(role_lrs[role_key])
    role_scales = extra.get("optimizer_role_lr_scales", {})
    if isinstance(role_scales, dict) and role_key in role_scales:
        return base_lr * float(role_scales[role_key])

    if role_key == "backbone" and getattr(spec, "backbone_lr", None) is not None:
        return float(getattr(spec, "backbone_lr"))
    if getattr(spec, "head_lr", None) is not None:
        return float(getattr(spec, "head_lr"))
    if isinstance(role_scales, dict) and "default" in role_scales:
        return base_lr * float(role_scales["default"])
    if isinstance(role_lrs, dict) and "default" in role_lrs:
        return float(role_lrs["default"])
    return base_lr


def _build_default_optimizer(model: Any, config: Any, policy: str) -> tuple[Any, dict[str, Any]]:
    """
    构建普通单组 AdamW。
    English: Build a standard single-group AdamW optimizer.
    """

    import torch

    lr = float(getattr(config, "LEARNING_RATE", 1e-4))
    weight_decay = float(getattr(config, "WEIGHT_DECAY", 1e-3))
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    plan = {
        "policy": policy,
        "group_specs": [
            {
                "name": "all",
                "lr_role": "default",
                "lr": lr,
                "initial_lr": lr,
                "weight_decay": weight_decay,
                "param_count": int(sum(param.numel() for param in params)),
                "params": params,
            }
        ],
        "freeze_backbone_epochs": 0,
        "uses_layerwise_lr": False,
    }
    optimizer.param_groups[0]["name"] = "all"
    optimizer.param_groups[0]["lr_role"] = "default"
    optimizer.param_groups[0]["initial_lr"] = lr
    return optimizer, plan


def build_optimizer_for_spec(model: Any, spec: Any, config: Any) -> tuple[Any, dict[str, Any]]:
    """
    根据菜单 ModelSpec 和模型参数分组构建 optimizer。
    English: Build the optimizer from the menu `ModelSpec` and model parameter groups.

    支持策略:
    English: :
        default_adamw: 所有可训练参数使用同一学习率；
        English: default_adamw: trainingparameterlearning rate;
        layerwise_lr: 按模型 `lr_role` 参数分组和菜单 role LR/倍率构建分组学习率；
        English: layerwise_lr: model `lr_role` parameter groupsmenu role LR/buildlearning rate;
        freeze_then_layerwise: 前 `freeze_backbone_epochs` 个 epoch 冻结 backbone，之后按 layerwise_lr 微调。
        English: freeze_then_layerwise: `freeze_backbone_epochs` epoch backbone, layerwise_lr .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    import torch

    policy = normalize_optimizer_policy(getattr(spec, "optimizer_policy", ""))
    if policy == DEFAULT_OPTIMIZER_POLICY:
        return _build_default_optimizer(model, config, policy)
    if policy not in LAYERWISE_POLICIES:
        raise ValueError(f"未知 optimizer_policy={policy!r}。")

    weight_decay = float(getattr(config, "WEIGHT_DECAY", 1e-3))
    freeze_backbone_epochs = int(getattr(spec, "freeze_backbone_epochs", 0) or 0)

    model_groups = collect_model_optimizer_groups(model)
    backbone_groups = [group for group in model_groups if str(group["lr_role"]) == "backbone"]
    other_groups = [group for group in model_groups if str(group["lr_role"]) != "backbone"]
    ordered_groups = other_groups + backbone_groups

    optimizer_groups = []
    group_specs = []
    for group in ordered_groups:
        role = group["lr_role"]
        lr = resolve_lr_for_role(role, spec, config)
        optimizer_groups.append({
            "params": group["params"],
            "lr": lr,
            "weight_decay": weight_decay,
            "name": group["name"],
            "lr_role": role,
            "initial_lr": lr,
        })
        group_specs.append({
            "name": group["name"],
            "lr_role": role,
            "lr": lr,
            "initial_lr": lr,
            "weight_decay": weight_decay,
            "param_count": int(group["param_count"]),
            "params": group["params"],
        })

    optimizer = torch.optim.AdamW(optimizer_groups)
    plan = {
        "policy": policy,
        "group_specs": group_specs,
        "freeze_backbone_epochs": freeze_backbone_epochs if policy == "freeze_then_layerwise" else 0,
        "uses_layerwise_lr": True,
        "head_lr": resolve_lr_for_role("head", spec, config),
        "backbone_lr": resolve_lr_for_role("backbone", spec, config),
        "role_lrs": {
            str(group["lr_role"]): float(resolve_lr_for_role(str(group["lr_role"]), spec, config))
            for group in ordered_groups
        },
    }
    return optimizer, plan


def sync_optimizer_policy_for_epoch(model: Any, optimizer_plan: dict[str, Any], epoch: int) -> dict[str, Any]:
    """
    按当前 epoch 同步冻结/解冻状态。
    English: current epoch /.

    说明:
    English: :
        `freeze_then_layerwise` 在 epoch <= freeze_backbone_epochs 时冻结 backbone；
        English: `freeze_then_layerwise` epoch <= freeze_backbone_epochs backbone;
        后续 epoch 自动解冻，optimizer 已包含 backbone 参数组，因此无需重建 optimizer。
        English: epoch , optimizer backbone parameter, optimizer.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    policy = str(optimizer_plan.get("policy", DEFAULT_OPTIMIZER_POLICY))
    freeze_until = int(optimizer_plan.get("freeze_backbone_epochs", 0) or 0)
    freeze_backbone = policy == "freeze_then_layerwise" and int(epoch) <= freeze_until
    frozen_param_count = 0
    trainable_param_count = 0
    for group in optimizer_plan.get("group_specs", []):
        should_train = not (freeze_backbone and str(group.get("lr_role")) == "backbone")
        for param in group.get("params", []):
            param.requires_grad = bool(should_train)
            if should_train:
                trainable_param_count += int(param.numel())
            else:
                frozen_param_count += int(param.numel())
    return {
        "optimizer_policy": policy,
        "freeze_backbone_epochs": freeze_until,
        "backbone_frozen_by_optimizer_policy": bool(freeze_backbone),
        "optimizer_policy_epoch": int(epoch),
        "optimizer_frozen_param_count": int(frozen_param_count),
        "optimizer_trainable_param_count": int(trainable_param_count),
    }


def get_optimizer_memory_preflight_epoch(optimizer_plan: dict[str, Any], start_epoch: int) -> int:
    """
    返回显存预检应模拟的 epoch。
    English: return epoch.

    对冻结预热策略，预检使用解冻后的第一个 epoch，避免只按冻结阶段估算显存导致后续解冻后超限。
    English: , epoch, avoid.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    freeze_until = int(optimizer_plan.get("freeze_backbone_epochs", 0) or 0)
    if str(optimizer_plan.get("policy")) == "freeze_then_layerwise" and freeze_until > 0:
        return max(int(start_epoch), freeze_until + 1)
    return int(start_epoch)


def serialize_optimizer_plan(optimizer_plan: dict[str, Any]) -> dict[str, Any]:
    """
    去掉参数对象，返回可写入 JSON/CSV 的 optimizer 策略摘要。
    English: parameter, returnwrite JSON/CSV optimizer policy.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    return {
        "policy": str(optimizer_plan.get("policy", DEFAULT_OPTIMIZER_POLICY)),
        "freeze_backbone_epochs": int(optimizer_plan.get("freeze_backbone_epochs", 0) or 0),
        "uses_layerwise_lr": bool(optimizer_plan.get("uses_layerwise_lr", False)),
        "head_lr": optimizer_plan.get("head_lr"),
        "backbone_lr": optimizer_plan.get("backbone_lr"),
        "role_lrs": {
            str(key): float(value)
            for key, value in dict(optimizer_plan.get("role_lrs", {}) or {}).items()
        },
        "groups": [
            {
                "name": str(group.get("name", "")),
                "lr_role": str(group.get("lr_role", "")),
                "initial_lr": float(group.get("initial_lr", group.get("lr", 0.0))),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "param_count": int(group.get("param_count", 0)),
            }
            for group in optimizer_plan.get("group_specs", [])
        ],
    }


def get_optimizer_lrs(optimizer: Any) -> dict[str, float]:
    """
    返回当前 optimizer 各参数组学习率。
    English: returncurrent optimizer parameterlearning rate.
    """

    values: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups, start=1):
        name = str(group.get("name") or f"group_{index}")
        values[name] = float(group["lr"])
    return values


def get_primary_optimizer_lr(optimizer: Any) -> float:
    """
    返回用于终端显示和旧字段兼容的主学习率。
    English: returnfieldcompatiblelearning rate.
    """

    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def restore_optimizer_lrs_from_progress(optimizer: Any, progress: dict[str, Any] | None) -> None:
    """
    从 training_progress.json 恢复分组学习率。
    English: training_progress.json learning rate.

    兼容逻辑:
    English: compatibleLogic:
    - 新输出优先读取 `optimizer_lrs`，逐组恢复；
    English: - Outputread `optimizer_lrs`, ;
    - 旧输出只有 `current_lr` 时，按 group initial_lr 等比例缩放，保留 backbone/head 比例。
    English: - Output `current_lr` , group initial_lr , backbone/head .

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    if not progress:
        return
    group_lrs = progress.get("optimizer_lrs")
    if isinstance(group_lrs, dict) and group_lrs:
        for index, group in enumerate(optimizer.param_groups, start=1):
            name = str(group.get("name") or f"group_{index}")
            if name in group_lrs:
                group["lr"] = float(group_lrs[name])
        return

    if progress.get("current_lr") is None or not optimizer.param_groups:
        return
    primary_initial_lr = float(optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]))
    if primary_initial_lr <= 0:
        return
    scale = float(progress["current_lr"]) / primary_initial_lr
    for group in optimizer.param_groups:
        group_initial_lr = float(group.get("initial_lr", group["lr"]))
        group["lr"] = group_initial_lr * scale


def decay_optimizer_lrs(optimizer: Any, factor: float) -> dict[str, float]:
    """
    按比例衰减所有参数组学习率，并返回衰减后的分组 LR。
    English: parameterlearning rate, return LR.
    """

    ratio = float(factor)
    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"]) * ratio
    return get_optimizer_lrs(optimizer)


__all__ = [
    "DEFAULT_OPTIMIZER_POLICY",
    "build_optimizer_for_spec",
    "decay_optimizer_lrs",
    "get_optimizer_lrs",
    "get_optimizer_memory_preflight_epoch",
    "get_primary_optimizer_lr",
    "normalize_optimizer_policy",
    "resolve_lr_for_role",
    "restore_optimizer_lrs_from_progress",
    "serialize_optimizer_plan",
    "sync_optimizer_policy_for_epoch",
]
