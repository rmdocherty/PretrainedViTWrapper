import pytest
import torch
from torch import nn

from PVW import (
    BackboneConfig,
    BackboneRegistry,
    PretrainedViTWrapper,
    WrapperConfig,
    WrapperRegistry,
)
from PVW.types import ViTBackbone


# Create a Mock Backbone class
class MockProj:
    def __init__(self):
        self.stride = (16, 16)


class MockPatchEmbed:
    def __init__(self):
        self.proj = MockProj()
        self.patch_size = (16, 16)


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 384
        self.patch_embed = MockPatchEmbed()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 384))


# Create a Mock Wrapper class (subclass of PretrainedViTWrapper)
class CustomWrapper(PretrainedViTWrapper):
    def __init__(self, vit: ViTBackbone, device, custom_arg="default"):
        super().__init__(vit, device=device)
        self.custom_arg = custom_arg


def test_custom_backbone_and_wrapper_registration():
    # 1. Register mock backbone builder in BackboneRegistry
    def build_mock_backbone(cfg: BackboneConfig) -> ViTBackbone:
        return MockBackbone()

    BackboneRegistry.register("mock_type", build_mock_backbone)

    # 2. Register mock wrapper configuration in ViTFactory
    custom_cfg = WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="mock_type",
            model_arch="dinov2_s",  # arbitrary arch name for testing
        ),
        wrapper_class=CustomWrapper,
        wrapper_kwargs={"custom_arg": "registered_value"},
    )

    WrapperRegistry.register("my_custom_model", custom_cfg)

    # 3. Build the model through ViTFactory
    model = WrapperRegistry.build("my_custom_model", device="cpu")

    # 4. Assertions to verify the architecture was successfully constructed
    assert isinstance(model, CustomWrapper)
    assert isinstance(model.vit, MockBackbone)
    assert model.custom_arg == "registered_value"
    assert model.device == "cpu"
    assert model.stride == (16, 16)
    assert model.patch_size == (16, 16)
    assert model.embed_dim == 384
    assert model.num_blocks == 6


def test_factory_arg_overrides():
    # Verify that we can override registered kwargs at build time
    model = WrapperRegistry.build("my_custom_model", device="cpu", custom_arg="overridden_value")
    assert isinstance(model, CustomWrapper)
    assert model.custom_arg == "overridden_value"


def test_autoregistration_invalid_name():
    # Attempting to build an unknown architecture that's not in ARCHS_TO_IDS should raise KeyError
    with pytest.raises(KeyError):
        WrapperRegistry.build("non_existent_arch", device="cpu")


def test_timm_backbone_building():
    # Build a standard model using default factory path with pretrained=False
    cfg = WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="timm",
            model_arch="deit_s",
            pretrained=False,
        )
    )
    WrapperRegistry.register("deit_s_random", cfg)
    model = WrapperRegistry.build("deit_s_random", device="cpu")
    assert isinstance(model, PretrainedViTWrapper)
    assert model.name == "deit_s_random"
    assert model.arch_name == "deit_s"
    assert model.embed_dim == 384
