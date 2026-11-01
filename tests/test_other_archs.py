# test timm SAM feature backbone, Deit, MAE etc

# SAM: has no prefix tokens and uses `forward_intermediates`
# EVA02 uses `forward_intermediates`

from PIL import Image, ImageDraw

from PVW import (
    BackboneConfig,
    BackboneRegistry,
    PretrainedViTWrapper,
    WrapperConfig,
    WrapperRegistry,
)
from PVW.types import (
    TimmStyleViTForwardIntermediates,
    TimmStyleViTGetIntermediates,
    satisfies_protocol,
)


def create_synthetic_image(size: tuple[int, int] = (300, 300)) -> Image.Image:
    img = Image.new("RGB", size, color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 250, 250], fill=(250, 100, 100), outline=(255, 255, 255))
    draw.rectangle([100, 100, 200, 200], fill=(100, 250, 100))
    return img


def test_sam_backbone_satisfies_protocol():
    cfg = BackboneConfig(
        backbone_type="timm_fwd",
        model_arch="sam_b",
        pretrained=False,
    )
    vit = BackboneRegistry.build(cfg)

    assert satisfies_protocol(vit, TimmStyleViTForwardIntermediates)
    assert not satisfies_protocol(vit, TimmStyleViTGetIntermediates, verbose=False)
    assert vit.num_prefix_tokens == 0


def test_eva02_backbone_satisfies_protocol():
    cfg = BackboneConfig(
        backbone_type="timm_fwd",
        model_arch="eva02_b",
        pretrained=False,
    )
    vit = BackboneRegistry.build(cfg)

    assert satisfies_protocol(vit, TimmStyleViTForwardIntermediates)
    assert not satisfies_protocol(vit, TimmStyleViTGetIntermediates, verbose=False)
    assert vit.num_prefix_tokens == 1


def test_sam_forward_features():
    cfg = WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="timm_fwd",
            model_arch="sam_b",
            pretrained=False,
        ),
    )
    WrapperRegistry.register("sam_b_test", cfg)
    model = WrapperRegistry.build("sam_b_test", device="cpu")

    assert isinstance(model, PretrainedViTWrapper)
    assert model.embed_dim == 256
    assert model.patch_size == (16, 16)
    assert model.num_prefix_tokens == 0

    # 300x300 -> closest multiple of 16 is 288x288 -> 18x18 patches
    pil_img = create_synthetic_image((300, 300))
    features_2d = model.forward_features(pil_img, make_2D=True)
    assert features_2d.shape == (1, 256, 18, 18)

    features_1d = model.forward_features(pil_img, make_2D=False)
    assert features_1d.shape == (1, 256, 18 * 18)


def test_sam_forward_intermediates():
    cfg = WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="timm_fwd",
            model_arch="sam_b",
            pretrained=False,
        ),
    )
    WrapperRegistry.register("sam_b_test", cfg)
    model = WrapperRegistry.build("sam_b_test", device="cpu")

    pil_img = create_synthetic_image((300, 300))
    intermediates_2d = model.forward_intermediates(pil_img, n=2, make_2D=True)

    assert isinstance(intermediates_2d, list)
    assert len(intermediates_2d) == 2
    for feat in intermediates_2d:
        assert feat.shape == (1, 256, 18, 18)

    intermediates_1d = model.forward_intermediates(pil_img, n=2, make_2D=False)
    assert isinstance(intermediates_1d, list)
    assert len(intermediates_1d) == 2
    for feat in intermediates_1d:
        assert feat.shape == (1, 256, 18 * 18)


def test_eva02_forward_features():
    cfg = WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="timm_fwd",
            model_arch="eva02_b",
            pretrained=False,
        ),
    )
    WrapperRegistry.register("eva02_b_test", cfg)
    model = WrapperRegistry.build("eva02_b_test", device="cpu")

    assert isinstance(model, PretrainedViTWrapper)
    assert model.embed_dim == 768
    assert model.patch_size == (14, 14)
    assert model.num_prefix_tokens == 1

    # 300x300 -> closest multiple of 14 is 294x294 -> 21x21 patches
    pil_img = create_synthetic_image((300, 300))
    features_2d = model.forward_features(pil_img, make_2D=True)
    assert features_2d.shape == (1, 768, 21, 21)

    features_1d = model.forward_features(pil_img, make_2D=False)
    assert features_1d.shape == (1, 768, 21 * 21)


def test_eva02_forward_intermediates():
    cfg = WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="timm_fwd",
            model_arch="eva02_b",
            pretrained=False,
        ),
    )
    WrapperRegistry.register("eva02_b_test", cfg)
    model = WrapperRegistry.build("eva02_b_test", device="cpu")

    pil_img = create_synthetic_image((300, 300))
    intermediates_2d = model.forward_intermediates(pil_img, n=2, make_2D=True)

    assert isinstance(intermediates_2d, list)
    assert len(intermediates_2d) == 2
    for feat in intermediates_2d:
        assert feat.shape == (1, 768, 21, 21)

    intermediates_1d = model.forward_intermediates(pil_img, n=2, make_2D=False)
    assert isinstance(intermediates_1d, list)
    assert len(intermediates_1d) == 2
    for feat in intermediates_1d:
        assert feat.shape == (1, 768, 21 * 21)


