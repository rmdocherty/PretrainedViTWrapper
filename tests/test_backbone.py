import os

import pytest
from torch import nn

from PVW import BackboneConfig, BackboneRegistry
from PVW.types import Dv3ViT, HasAttn, HasBlock, TimmViT, satisfies_protocol


def test_satisfies_protocol_non_vit():
    # 1. Test basic PyTorch modules that are not ViTs
    linear = nn.Linear(10, 10)
    conv = nn.Conv2d(3, 64, 3)
    basic_module = nn.Module()

    # None of these should satisfy the ViTBackbone protocol
    assert not satisfies_protocol(linear, TimmViT)
    assert not satisfies_protocol(conv, TimmViT)
    assert not satisfies_protocol(basic_module, TimmViT)


def test_satisfies_protocol_vit():
    # 2. Build a real ViT backbone using the library factory
    cfg = BackboneConfig(
        backbone_type="timm",
        model_arch="deit_s",
        pretrained=False,
    )
    vit = BackboneRegistry.build(cfg)

    # The created backbone should satisfy the ViTBackbone protocol
    assert satisfies_protocol(vit, TimmViT)


def test_satisfies_protocol_blocks_and_attn():
    # 3. Build a standard timm ViT to extract blocks and attention modules
    cfg = BackboneConfig(
        backbone_type="timm",
        model_arch="deit_s",
        pretrained=False,
    )
    vit = BackboneRegistry.build(cfg)

    # Extract block and attention modules
    block = vit.blocks[0]
    attn = block.attn

    # Test the structural protocols
    assert satisfies_protocol(block, HasBlock)
    assert not satisfies_protocol(block, HasAttn)
    assert satisfies_protocol(attn, HasAttn)


def test_satisfies_protocol_invalid_protocol_cls():
    # 4. Test that passing a non-protocol class raises TypeError
    linear = nn.Linear(10, 10)
    with pytest.raises(TypeError):
        satisfies_protocol(linear, nn.Linear)  # type: ignore


# Checkpoints and path checks for local loading tests
EUPE_CKPT = "models/checkpoints/EUPE-ViT-T.pt"

DINOV3_DIR_MISSING = not os.path.isdir("models/dinov3")
EUPE_DIR_MISSING = not os.path.isdir("models/EUPE")
EUPE_CKPT_MISSING = EUPE_DIR_MISSING or not os.path.exists(EUPE_CKPT)


@pytest.mark.skipif(DINOV3_DIR_MISSING, reason="models/dinov3 directory is missing")
def test_load_local_dinov3_no_weights():
    cfg = BackboneConfig(
        backbone_type="torch_hub",
        model_arch="dinov3_s+",
        model_conf_path="models/dinov3",
        checkpoint_path=None,
    )
    vit = BackboneRegistry.build(cfg)
    assert satisfies_protocol(vit, Dv3ViT)


@pytest.mark.skipif(EUPE_DIR_MISSING, reason="models/EUPE directory is missing")
def test_load_local_eupe_no_weights():
    cfg = BackboneConfig(
        backbone_type="torch_hub",
        model_arch="EUPE_t",
        model_conf_path="models/EUPE",
        checkpoint_path=None,
    )
    vit = BackboneRegistry.build(cfg)
    assert satisfies_protocol(vit, Dv3ViT)


@pytest.mark.skipif(EUPE_CKPT_MISSING, reason="models/EUPE or checkpoint is missing")
def test_load_local_eupe_with_weights():
    cfg = BackboneConfig(
        backbone_type="torch_hub",
        model_arch="EUPE_t",
        model_conf_path="models/EUPE",
        checkpoint_path=EUPE_CKPT,
    )
    vit = BackboneRegistry.build(cfg)
    assert satisfies_protocol(vit, Dv3ViT)


@pytest.mark.skipif(DINOV3_DIR_MISSING, reason="models/dinov3 directory is missing")
def test_load_local_dinov3_remove_pos_embed():
    cfg = BackboneConfig(
        backbone_type="torch_hub",
        model_arch="dinov3_s+",
        model_conf_path="models/dinov3",
        checkpoint_path=None,
        remove_pos_embed=True,
    )
    vit = BackboneRegistry.build(cfg)
    assert vit.rope_embed is None


@pytest.mark.skipif(EUPE_DIR_MISSING, reason="models/EUPE directory is missing")
def test_load_local_eupe_remove_pos_embed():
    cfg = BackboneConfig(
        backbone_type="torch_hub",
        model_arch="EUPE_t",
        model_conf_path="models/EUPE",
        checkpoint_path=None,
        remove_pos_embed=True,
    )
    vit = BackboneRegistry.build(cfg)
    assert vit.rope_embed is None
