from types import MethodType

from torch import Tensor
from torch.nn import Parameter, Sequential

from .types import (
    Dv3StyleViT,
    HasBlock,
    HasPatchEmbed,
    TimmStyleViTGetIntermediates,
    V,
    satisfies_protocol,
)
from .utils import logger


def change_stride(vit: V, new_stride: tuple[int, int]) -> V:
    """Change the stride of the patch embedding layer of a ViT backbone.

    Args:
        vit (V): The ViT backbone to modify.
        new_stride (tuple[int, int]): The new stride to set for the patch embedding layer.
    """
    old_sx, old_sy = vit.patch_embed.proj.stride
    new_sx, new_sy = new_stride
    if (old_sx, old_sy) == (new_sx, new_sy):
        return vit

    vit.patch_embed.proj.stride = new_stride

    def dynamic_feat_size(self: HasPatchEmbed, img_size: tuple[int, int]) -> tuple[int, int]:
        """Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        return (img_size[0] - self.patch_size[0]) // self.proj.stride[0] + 1, (
            img_size[1] - self.patch_size[1]
        ) // self.proj.stride[1] + 1

    vit.patch_embed.dynamic_feat_size = MethodType(dynamic_feat_size, vit.patch_embed)  # type: ignore
    logger.info(f"Stride changed: {old_sx, old_sy} -> {new_sx, new_sy}")
    return vit


def replace_pos_embed(vit: V, new_pos_embed: Tensor | None, requires_grad: bool) -> V:
    """Replace the positional embedding of a ViT backbone.

    Args:
        vit (V): The ViT backbone to modify.
        new_pos_embed (Tensor): The new positional embedding to set.
    """
    if satisfies_protocol(vit, TimmStyleViTGetIntermediates):
        if isinstance(new_pos_embed, Tensor):
            vit.pos_embed = Parameter(new_pos_embed, requires_grad=requires_grad)
        else:
            logger.info("Removed default pos. embed")
            vit.pos_embed = None
    elif satisfies_protocol(vit, Dv3StyleViT):
        vit.rope_embed = None
        logger.info("Removed RoPE pos. embed")
    return vit


def replace_blocks(vit: V, new_blocks: list[HasBlock]) -> V:
    """Replace the blocks of a ViT backbone.

    Args:
        vit (V): The ViT backbone to modify.
        new_blocks (Tensor): The new blocks to set.
    """
    vit.blocks = Sequential(*new_blocks)  # type: ignore
    logger.info(f"Replaced blocks with {len(new_blocks)} {new_blocks[0].__class__.__name__}")
    return vit
