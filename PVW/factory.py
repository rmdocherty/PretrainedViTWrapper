from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from functools import partial
from inspect import Parameter, signature
from typing import Any, Literal

import torch
from timm import create_model

from PVW.types import TimmStyleViTForwardIntermediates

from .modifications import change_stride, replace_pos_embed
from .types import (
    ARCHS_TO_LOCAL_IDS,
    ARCHS_TO_TIMM_IDS,
    ARCHS_TO_TIMMFWD_IDS,
    Architectures,
    Dv3StyleViT,
    ImageTransform,
    ModificationFn,
    TimmStyleViTGetIntermediates,
    ViTBackbone,
    satisfies_protocol,
)
from .utils import closest_resize, logger
from .wrapper import PretrainedViTWrapper

Dtype = Literal["float32", "float16"]


@dataclass
class BackboneConfig:
    backbone_type: str = "timm"  # e.g., "timm", "dinov3", or a custom registered type
    model_arch: Architectures = "dinov2_s"
    pretrained: bool = True
    checkpoint_path: str | None = None
    model_conf_path: str | None = None

    stride: tuple[int, int] | None = None
    remove_pos_embed: bool = False
    add_flash_attn: bool = False

    dynamic_img_size: bool = True
    dynamic_img_pad: bool = False

    modifications: list[ModificationFn] = field(default_factory=list)
    dtype: torch.dtype = torch.float32


@dataclass
class WrapperConfig:
    backbone_cfg: BackboneConfig
    wrapper_class: type[PretrainedViTWrapper] = PretrainedViTWrapper
    wrapper_kwargs: dict[str, Any] = field(default_factory=dict)
    transform: ImageTransform = closest_resize
    dtype: torch.dtype = torch.float32


def _populate_modifications(
    prev_modifications: list[ModificationFn] | None, cfg: BackboneConfig
) -> list[ModificationFn]:
    if prev_modifications is None:
        prev_modifications = []
    else:
        # copy list to avoid mutating class default
        prev_modifications = list(prev_modifications)

    if cfg.stride:
        prev_modifications.append(partial(change_stride, new_stride=cfg.stride))

    if cfg.remove_pos_embed:
        prev_modifications.append(partial(replace_pos_embed, new_pos_embed=None, requires_grad=False))

    return prev_modifications


def _maybe_apply_mods_and_load_checkpoint(vit: ViTBackbone, cfg: BackboneConfig) -> ViTBackbone:
    # load checkpoints *after* modifications as some checkpoints don't have same structure e.g removed pos embeds
    mods = _populate_modifications(cfg.modifications, cfg)
    for mod in mods:
        vit = mod(vit)

    if cfg.checkpoint_path:
        logger.info(f"Loading checkpoint: {cfg.checkpoint_path}")
        state_dict = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
        vit.load_state_dict(state_dict, strict=True)

    return vit


def create_timmget_backbone(cfg: BackboneConfig) -> TimmStyleViTGetIntermediates:
    model_identifier = ARCHS_TO_TIMM_IDS[cfg.model_arch]
    if "sam" in model_identifier or "conv" in model_identifier:
        vit = create_model(
            model_identifier,
            pretrained=cfg.pretrained,
        )
    else:
        vit = create_model(
            model_identifier,
            pretrained=cfg.pretrained,
            dynamic_img_size=cfg.dynamic_img_size,
            dynamic_img_pad=cfg.dynamic_img_pad,
            pretrained_strict=False,
        )
    assert satisfies_protocol(vit, TimmStyleViTGetIntermediates)

    vit = _maybe_apply_mods_and_load_checkpoint(vit, cfg)
    vit = vit.to(dtype=cfg.dtype)

    return vit

def create_timmfwd_backbone(cfg: BackboneConfig) -> TimmStyleViTForwardIntermediates:
    model_identifier = ARCHS_TO_TIMMFWD_IDS[cfg.model_arch]
    if "sam" in model_identifier or "conv" in model_identifier:
        vit = create_model(
            model_identifier,
            pretrained=cfg.pretrained,
        )
        vit.num_prefix_tokens = 0
    else:
        vit = create_model(
            model_identifier,
            pretrained=cfg.pretrained,
            dynamic_img_size=cfg.dynamic_img_size,
            dynamic_img_pad=cfg.dynamic_img_pad,
            pretrained_strict=False,
        )
    assert satisfies_protocol(vit, TimmStyleViTForwardIntermediates)

    vit = _maybe_apply_mods_and_load_checkpoint(vit, cfg)
    vit = vit.to(dtype=cfg.dtype)

    return vit


def create_local_backbone(cfg: BackboneConfig) -> Dv3StyleViT:
    model_identifier = ARCHS_TO_LOCAL_IDS[cfg.model_arch]

    vit = torch.hub.load(cfg.model_conf_path, model_identifier, source="local", pretrained=cfg.pretrained)
    assert satisfies_protocol(vit, Dv3StyleViT)

    vit = _maybe_apply_mods_and_load_checkpoint(vit, cfg)
    vit = vit.to(dtype=cfg.dtype)

    return vit


class BackboneRegistry:
    _registry: dict[str, Callable[[BackboneConfig], ViTBackbone]] = {}

    @classmethod
    def register(cls, backbone_type: str, builder_fn: Callable[[BackboneConfig], ViTBackbone]) -> None:
        cls._registry[backbone_type] = builder_fn

    @classmethod
    def build(cls, cfg: BackboneConfig) -> ViTBackbone:
        logger.info("Building backbone with config: %s", cfg)
        builder = cls._registry.get(cfg.backbone_type)
        if not builder:
            raise ValueError(f"Backbone type '{cfg.backbone_type}' not registered in BackboneRegistry.")
        return builder(cfg)


# Register the default timm backbone builder
BackboneRegistry.register("timm", create_timmget_backbone)
BackboneRegistry.register("timm_fwd", create_timmfwd_backbone)
BackboneRegistry.register("torch_hub", create_local_backbone)


class WrapperRegistry:
    _registry: dict[str, WrapperConfig] = {}

    @classmethod
    def register(cls, name: str, cfg: WrapperConfig) -> None:
        cls._registry[name] = cfg

    @classmethod
    def build(cls, name: str, device: torch.device | str = "cuda", **override_kwargs) -> PretrainedViTWrapper:
        logger.info("Building wrapper '%s' on device %s", name, device)
        # Auto-register standard architectures if they are not in the registry but are valid timm architectures
        if name not in cls._registry:
            if name in ARCHS_TO_TIMM_IDS:
                cls.register(
                    name,
                    WrapperConfig(
                        backbone_cfg=BackboneConfig(backbone_type="timm", model_arch=name)  # type: ignore
                    ),
                )
            elif name in ARCHS_TO_TIMMFWD_IDS:
                cls.register(
                    name,
                    WrapperConfig(
                        backbone_cfg=BackboneConfig(backbone_type="timm_fwd", model_arch=name)  # type: ignore
                    ),
                )
            else:
                raise KeyError(f"Model variant '{name}' not found in registry and is not a default TimmArchitecture.")

        cfg = cls._registry[name]

        backbone_kwargs = {}
        for f in fields(cfg.backbone_cfg):
            if f.name in override_kwargs:
                backbone_kwargs[f.name] = override_kwargs.pop(f.name)

        if backbone_kwargs:
            backbone_cfg = replace(cfg.backbone_cfg, **backbone_kwargs)
        else:
            backbone_cfg = cfg.backbone_cfg

        # Build the backbone using BackboneRegistry
        vit = BackboneRegistry.build(backbone_cfg)

        # Prepare wrapper parameters
        wrapper_cls = cfg.wrapper_class
        kwargs = {
            **cfg.wrapper_kwargs,
            "device": device,
            **override_kwargs,
        }
        sig = signature(wrapper_cls.__init__)
        if "dtype" in sig.parameters or any(p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            kwargs["dtype"] = cfg.dtype

        transform = kwargs.pop("transform", cfg.transform)

        # Construct wrapper
        wrapper = wrapper_cls(vit=vit, **kwargs)
        wrapper.name = name
        wrapper.arch_name = cfg.backbone_cfg.model_arch
        wrapper.transform = transform
        return wrapper
