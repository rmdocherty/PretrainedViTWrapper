import logging
from collections.abc import Callable
from typing import (
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
)

from torch import Tensor
from torch.nn import Conv2d, Dropout, Linear, Module, Parameter, Sequential
from torchvision.transforms import Compose

P = ParamSpec("P")
ImageTransform = Callable[Concatenate[int, int, P], Compose]


"""
These are interfaces defining relevant ViT properties we want for type safety in our wrapper & modification code.
Should be respected by `timm` and DINO style models.  
"""

T = TypeVar("T")
logger = logging.getLogger(__name__)


def satisfies_protocol(obj: object, protocol_cls: type[T], verbose: bool = True) -> TypeGuard[T]:
    """Runtime check if an object implements the structural interface of a Protocol class."""
    protocol_attrs = getattr(protocol_cls, "__protocol_attrs__", None)
    if protocol_attrs is None:
        raise TypeError(f"{protocol_cls} is not a Protocol class")
    attrs = cast(tuple[str, ...], protocol_attrs)
    missing_attrs = [attr for attr in attrs if not hasattr(obj, attr)]
    if missing_attrs:
        print(dir(obj))
        if verbose:
            logger.warning(
                "%s doesn't satisfy %s; missing: %s",
                type(obj).__name__,
                protocol_cls.__name__,
                ", ".join(missing_attrs),
            )
        return False
    return True


class HasAttn(Protocol):
    num_heads: int
    head_dim: int
    attn_dim: int
    scale: float
    fused_attn: bool

    qkv: Linear
    q_norm: Module
    k_norm: Module
    attn_drop: Dropout
    norm: Module
    proj: Linear
    proj_drop: Dropout

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor: ...


class HasBlock(Protocol):
    norm1: Module
    attn: HasAttn
    ls1: Module
    drop_path1: Module

    norm2: Module
    mlp: Module
    ls2: Module
    drop_path2: Module

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor: ...


class HasPatchEmbed(Protocol):
    patch_size: tuple[int, int]
    proj: Conv2d

    def dynamic_feat_size(self, img_size: tuple[int, int]) -> tuple[int, int]: ...


class BaseViTMixin(Protocol):
    num_features: int
    patch_embed: HasPatchEmbed
    blocks: Sequential | Module
    state_dict: dict[str, Any]

    def __call__(self, x: Tensor, *args: Any, **kwds: Any) -> Tensor: ...

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any: ...


class TimmStyleViTGetIntermediates(BaseViTMixin, Protocol):
    # Model constants
    num_features: int

    num_prefix_tokens: int

    # Layers
    pos_embed: Parameter | None

    # Methods
    def forward_features(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor: ...

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: int | list[int] | tuple[int] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        attn_mask: Tensor | None = None,
    ) -> list[Tensor]: ...

    def to(self, *args: Any, **kwargs: Any) -> "TimmStyleViTGetIntermediates": ...


class TimmStyleViTForwardIntermediates(BaseViTMixin, Protocol):
    # Model constants
    num_features: int

    num_prefix_tokens: int

    # Layers
    pos_embed: Parameter | None

    # Methods
    def forward_features(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor: ...

    def forward_intermediates(
        self,
        x: Tensor,
        indices: int | list[int] | tuple[int] | None = None,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = 'NCHW',
        intermediates_only: bool = False,
    ) -> list[Tensor]: ...

    def to(self, *args: Any, **kwargs: Any) -> "TimmStyleViTForwardIntermediates": ...




class Dv3StyleViT(BaseViTMixin, Protocol):
    # Model constants
    n_storage_tokens: int

    # Layers
    rope_embed: Parameter | None

    # Methods
    def forward_features(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> dict[str, Tensor]: ...

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: int | list[int] | tuple[int] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = False,
    ) -> tuple[Tensor | tuple[Tensor, ...]]: ...

    def to(self, *args: Any, **kwargs: Any) -> "Dv3StyleViT": ...


ViTBackbone = TimmStyleViTGetIntermediates | TimmStyleViTForwardIntermediates | Dv3StyleViT

V = TypeVar("V", bound=ViTBackbone)
J = ParamSpec("J")
ModificationFn = Callable[Concatenate[V, J], V]

Architectures = Literal[
    "dino_s",
    "dino_b",
    "dinov2_s",
    "dinov2_b",
    "dinov2_l",
    "dinov3_s",
    "dinov3_s+",
    "dinov3_b",
    "dinov3_l",
    "EUPE_t",
    "EUPE_s",
    "mae_b",
    "in1k_b",
    "deit_s",
    "sam_b",
    "clip_b",
    "eva02_b",
    "convnext",
    "vit_t_in",
]
ARCHS_TO_TIMM_IDS: dict[Architectures, str] = {
    "dino_s": "vit_small_patch16_224.dino",
    "dino_b": "vit_base_patch16_224.dino",
    "dinov2_s": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_b": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov2_l": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov3_s": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_s+": "vit_small_plus_patch16_dinov3.lvd1689m",
    "dinov3_b": "vit_base_patch16_dinov3.lvd1689m",
    "EUPE_t": "hf_hub:timm/vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m",
    "EUPE_s": "hf_hub:timm/vit_small_patch16_dinov3_qkvb.eupe_lvd1689m",
    "mae_b": "vit_base_patch16_224.mae",
    "in1k_b": "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
    "deit_s": "deit3_small_patch16_224.fb_in1k",
    "clip_b": "vit_base_patch16_clip_224.openai",
    "convnext": "convnextv2_tiny.fcmae",
    "vit_t_in": "tiny_vit_5m_224.dist_in22k_ft_in1k",
}
ARCHS_TO_TIMMFWD_IDS: dict[Architectures, str] = {
    "sam_b": "samvit_base_patch16.sa1b",
    "eva02_b": "eva02_base_patch14_224.mim_in22k",
}
ARCHS_TO_LOCAL_IDS: dict[Architectures, str] = {
    "dinov3_s": "dinov3_vits16",
    "dinov3_s+": "dinov3_vits16plus",
    "dinov3_b": "dinov3_vitb16",
    "dinov3_l": "dinov3_vitl16",
    "EUPE_t": "eupe_vitt16",
    "EUPE_s": "eupe_vits16",
}
