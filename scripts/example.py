import matplotlib.pyplot as plt
from PIL import Image
from torch import no_grad

from PVW import BackboneConfig, BackboneRegistry, PretrainedViTWrapper
from PVW.utils import do_pca, to_numpy

# backbone_cfg = BackboneConfig(
#     backbone_type="torch_hub",
#     model_arch="EUPE_s",
#     model_conf_path="models/EUPE",
#     checkpoint_path="../dino-saw/trained_models/EUPE-ViT-S.pt",
# )
# backbone_cfg = BackboneConfig(
#     backbone_type="torch_hub",
#     model_arch="dinov3_s+",
#     model_conf_path="models/dinov3",
#     checkpoint_path="../dino-saw/trained_models/dinov3_vits_patch16_plus_reg4.pth",
# )

# backbone_cfg = BackboneConfig(
#     backbone_type="timm",
#     model_arch="sam_b",
#     # model_conf_path="models/EUPE",
#     # checkpoint_path="../dino-saw/trained_models/EUPE-ViT-S.pt",
#     pretrained=False
# )
backbone_cfg = BackboneConfig(
    backbone_type="timm",
    model_arch="convnext",
    # model_conf_path="models/EUPE",
    # checkpoint_path="../dino-saw/trained_models/EUPE-ViT-S.pt",
    pretrained=False
)
backbone = BackboneRegistry.build(backbone_cfg)

pvt = PretrainedViTWrapper(backbone, device="cuda", name="SAMB", arch_name="dinov3_s")

img = Image.open("tests/data/ni_superalloy.png").convert("RGB")

with no_grad():
    feats = pvt.forward_features(img, True)

feats_reduced = do_pca(feats, 9)
feats_reduced_np = to_numpy(feats_reduced)

plt.imsave("tests/out/test.png", feats_reduced_np.transpose(1, 2, 0)[:, :, 0:3])
