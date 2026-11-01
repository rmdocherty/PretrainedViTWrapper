import os

# Set matplotlib backend to non-interactive 'Agg' before importing pyplot
import matplotlib
import pytest
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PVW import (
    BackboneConfig,
    BackboneRegistry,
    PretrainedViTWrapper,
    WrapperConfig,
    WrapperRegistry,
    do_pca,
)
from PVW.utils import closest_resize


def create_synthetic_image() -> Image.Image:
    # Create a colorful synthetic image with shapes
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    # Draw a red circle
    draw.ellipse([50, 50, 250, 250], fill=(250, 100, 100), outline=(255, 255, 255))
    # Draw a green square
    draw.rectangle([100, 100, 200, 200], fill=(100, 250, 100))
    return img


def test_dinov2_wrapper_with_pil_and_pca():
    # 1. Define transform
    img_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # 2. Register and build dinov2_s with pretrained=False (offline-friendly)
    cfg = WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="timm",
            model_arch="dinov2_s",
            pretrained=False,
        ),
        transform=lambda h, w, **kwargs: img_transform,
    )
    WrapperRegistry.register("dinov2_s_test", cfg)

    model = WrapperRegistry.build("dinov2_s_test", device="cpu")
    assert isinstance(model, PretrainedViTWrapper)
    assert model.transform is not None

    # 3. Create synthetic image and feed into forward_features
    pil_img = create_synthetic_image()
    features = model.forward_features(pil_img, make_2D=True)

    # 4. Check feature shapes
    # dinov2_s has embed_dim=384, patch_size=14. Input size is 224x224.
    # tokens = 224 / 14 = 16. Shape should be (1, 384, 16, 16).
    assert features.shape == (1, 384, 16, 16)

    # 5. Run PCA to reduce channels to 3 (RGB)
    pca_features = do_pca(features, n_components=3)
    assert pca_features.shape == (1, 3, 16, 16)

    # 6. Save visualization with matplotlib
    pca_np = pca_features.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # Scale pca_np to [0, 1] range for plotting
    pca_np = (pca_np - pca_np.min()) / (pca_np.max() - pca_np.min() + 1e-8)

    _fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pil_img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(pca_np)
    axes[1].set_title("DINOv2 PCA Features")
    axes[1].axis("off")

    plt.tight_layout()

    # Save to the conversation artifact directory
    artifact_dir = "/home/ronan/.gemini/antigravity-cli/brain/8b606ad2-b139-4e11-85cb-9dbdacce5d69"
    os.makedirs(artifact_dir, exist_ok=True)
    save_path = os.path.join(artifact_dir, "pca_visualization.png")

    plt.savefig(save_path, dpi=150)
    plt.close()

    assert os.path.exists(save_path)


def test_dinov2_size_agnostic_transform():
    # 1. Register and build dinov2_s with closest_resize as the size-agnostic transform creator
    cfg = WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="timm",
            model_arch="dinov2_s",
            pretrained=False,
        ),
        transform=closest_resize,
    )
    WrapperRegistry.register("dinov2_s_size_agnostic", cfg)

    model = WrapperRegistry.build("dinov2_s_size_agnostic", device="cpu")
    assert isinstance(model, PretrainedViTWrapper)
    assert model.transform == closest_resize

    # 2. Create 300x300 image and run forward features
    pil_img = create_synthetic_image()  # 300x300
    features = model.forward_features(pil_img, make_2D=True)

    # 3. Check shape
    # 300 % 14 = 6. Nearest lower multiple is 300 - 6 = 294.
    # tokens per dimension = 294 / 14 = 21.
    # Expected feature shape: (1, 384, 21, 21)
    assert features.shape == (1, 384, 21, 21)


def test_dinov2_forward_intermediates():
    # 1. Build the registered "dinov2_s_size_agnostic" model
    model = WrapperRegistry.build("dinov2_s_size_agnostic", device="cpu")
    assert isinstance(model, PretrainedViTWrapper)

    # 2. Run forward_intermediates for the last 2 layers
    pil_img = create_synthetic_image()  # 300x300
    intermediates = model.forward_intermediates(pil_img, n=2, make_2D=True)

    # We requested n=2 layers, so we should get a list of 2 tensors
    assert isinstance(intermediates, list)
    assert len(intermediates) == 2
    for feat in intermediates:
        # Expected shape: (1, 384, 21, 21)
        assert feat.shape == (1, 384, 21, 21)


DINOV3_DIR_MISSING = not os.path.isdir("models/dinov3")


@pytest.mark.skipif(DINOV3_DIR_MISSING, reason="models/dinov3 directory is missing")
def test_dinov3_forward_features_and_intermediates():
    # Build unweighted DINOv3 backbone
    cfg = BackboneConfig(
        backbone_type="torch_hub",
        model_arch="dinov3_s+",
        model_conf_path="models/dinov3",
        checkpoint_path=None,
    )
    vit = BackboneRegistry.build(cfg)
    model = PretrainedViTWrapper(vit, device="cpu", name="dinov3_s", arch_name="dinov3_s+")

    # Create synthetic input image (300x300)
    pil_img = create_synthetic_image()

    # Test forward_features
    # dinov3_s+ has embed_dim=384, patch_size=16
    # 300 % 16 = 12. 300 - 12 = 288.
    # tokens per dimension = 288 / 16 = 18.
    # Expected shape: (1, 384, 18, 18)
    features = model.forward_features(pil_img, make_2D=True)
    assert features.shape == (1, 384, 18, 18)

    # Test forward_intermediates
    intermediates = model.forward_intermediates(pil_img, n=2, make_2D=True)
    assert isinstance(intermediates, list)
    assert len(intermediates) == 2
    for feat in intermediates:
        assert feat.shape == (1, 384, 18, 18)


def test_wrapper_dtype_casting():
    # 1. Register and build a model with a custom dtype (float16)
    cfg = WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="timm",
            model_arch="dinov2_s",
            pretrained=False,
            dtype=torch.float16,
        ),
        dtype=torch.float16,
    )
    WrapperRegistry.register("dinov2_s_float16_test", cfg)

    model = WrapperRegistry.build("dinov2_s_float16_test", device="cpu")
    assert isinstance(model, PretrainedViTWrapper)
    assert model.dtype == torch.float16

    # Check that backbone parameters were cast to float16
    for param in model.vit.parameters():
        assert param.dtype == torch.float16
        break

    # 2. Run preprocess_input with a PIL image and check output dtype is float16
    pil_img = create_synthetic_image()
    tensor = model.preprocess_input(pil_img)
    assert tensor.dtype == torch.float16

    # 3. Test forward pass shapes and dtype
    features = model.forward_features(pil_img, make_2D=True)
    assert features.dtype == torch.float16

