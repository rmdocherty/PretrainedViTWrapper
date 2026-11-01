import torch
from PIL import Image
from torch import Tensor
from torch.nn import Module

from .types import (
    Architectures,
    Dv3StyleViT,
    ImageTransform,
    TimmStyleViTGetIntermediates,
    ViTBackbone,
    satisfies_protocol,
)
from .utils import closest_resize, logger


class PretrainedViTWrapper(Module):
    def __init__(
        self,
        vit: ViTBackbone,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        transform: ImageTransform = closest_resize,
        name: str = "",
        arch_name: Architectures | None = None,
    ):
        """Wrapper around a ViTBackone (timm or local) with unified typesafe interface.
        Designed for feature extraction of variable sized `x`.


        Args:
            vit (ViTBackbone): pytorch module that satisfies given ViTBackbone protocol (timm or local)
            device (_type_): torch device
            name (str, optional): _description_. Defaults to "".
            arch_name (Architectures | None, optional): _description_. Defaults to None.
            transform (ImageTransform, optional): transform generator i.e function that returns torchvision transform
                as a function of shape of `x` . Defaults to closest_resize.
        """
        super().__init__()
        self.vit: ViTBackbone = vit

        self.name: str = name
        self.arch_name: Architectures | None = arch_name
        self.device = device
        self.transform = transform
        self.dtype = dtype

        self.stride = self.vit.patch_embed.proj.stride
        self.patch_size = self.vit.patch_embed.patch_size

        self.embed_dim = self.vit.num_features
        self.num_blocks = len(self.vit.blocks)  # type: ignore

        self.to(device=device, dtype=self.dtype)
        logger.info(
            "Initialized PretrainedViTWrapper - name: '%s', arch: '%s', device: %s, stride: %s, patch_size: %s, embed_dim: %d, num_blocks: %d",
            self.name,
            self.arch_name,
            self.device,
            self.stride,
            self.patch_size,
            self.embed_dim,
            self.num_blocks,
        )

    @property
    def num_prefix_tokens(self) -> int:
        if satisfies_protocol(self.vit, TimmStyleViTGetIntermediates):
            return self.vit.num_prefix_tokens
        elif satisfies_protocol(self.vit, Dv3StyleViT):
            return self.vit.n_storage_tokens
        else:
            return 0

    def _input_tensor_checks(self, x: Tensor) -> bool:
        is_4D = x.ndim == 4
        is_h_patch_size_multiple = x.shape[2] % self.patch_size[0] == 0
        is_w_patch_size_multiple = x.shape[3] % self.patch_size[1] == 0
        return is_4D and is_h_patch_size_multiple and is_w_patch_size_multiple

    def preprocess_input(self, x: Tensor | Image.Image | list[Image.Image]) -> Tensor:
        """Apply input transformation to `x` iff `x` is a PIL image.
        NB: this 'transform' is actually a function that generates the correct transform based on dims of `x`

        Args:
            x (Tensor | Image.Image | list[Image.Image]): input image

        Raises:
            TypeError: x is invalid type

        Returns:
            Tensor: [B,C,H,W] tensor for `x`,
        """
        p_size = self.patch_size[0] if isinstance(self.patch_size, tuple) else self.patch_size
        if isinstance(x, Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        elif isinstance(x, Image.Image):
            w, h = x.size
            logger.info("Processing image, size: [%d, %d]", w, h)
            resolved_transform = self.transform(h, w, patch_size=p_size)
            img_tensor: Tensor = resolved_transform(x).unsqueeze(0).to(device=self.device, dtype=self.dtype)  # type: ignore
            assert self._input_tensor_checks(img_tensor), (
                f"`x` shape :{img_tensor.shape} not valid for patch size {self.patch_size}"
            )
            return img_tensor
        elif isinstance(x, list) and len(x) > 0 and isinstance(x[0], Image.Image):
            w, h = x[0].size
            logger.info("Processing %d images, size: [%d, %d]", len(x), w, h)
            resolved_transform = self.transform(h, w, patch_size=p_size)
            stacked = torch.stack([resolved_transform(img) for img in x]).to(device=self.device, dtype=self.dtype)  # type: ignore
            assert self._input_tensor_checks(stacked), (
                f"`x` shape :{stacked.shape} not valid for patch size {self.patch_size}"
            )
            return stacked

        raise TypeError(f"Unsupported input type: {type(x)}")

    def _get_tokens_per_dim(self, x: Tensor) -> tuple[int, int]:
        _b, _, h, w = x.shape
        p = self.patch_size
        s = self.stride
        n_patch_h, n_patch_w = (h - p[0]) // s[0] + 1, (w - p[1]) // s[1] + 1
        return n_patch_h, n_patch_w

    def forward(self, x: Tensor | Image.Image | list[Image.Image]) -> Tensor:
        x_tensor = self.preprocess_input(x)
        out = self.vit(x_tensor)
        logger.info(
            "Forward: x: [%s] -> f: [%s]",
            ",".join(map(str, x_tensor.shape)),
            ",".join(map(str, out.shape)),
        )
        return out

    def forward_intermediates(
        self,
        x: Tensor | Image.Image | list[Image.Image],
        n: int | list[int] = 1,
        make_2D: bool = True,
        norm: bool = True,
    ) -> list[Tensor]:
        x_tensor = self.preprocess_input(x)
        b, _, _, _ = x_tensor.shape
        n_patch_h, n_patch_w = self._get_tokens_per_dim(x_tensor)

        if satisfies_protocol(self.vit, TimmStyleViTGetIntermediates) or satisfies_protocol(self.vit, Dv3StyleViT):
            feats = self.vit.get_intermediate_layers(x_tensor, n=n, reshape=False, norm=norm)
        else:
            raise TypeError(f"Unsupported ViT type: {type(self.vit)}")

        feats = [f.permute(0, 2, 1) for f in feats]

        if make_2D:
            feats = [f.view(b, self.embed_dim, n_patch_h, n_patch_w) for f in feats]

        logger.info(
            "Forward Intermediates: x: [%s] -> f: %s",
            ",".join(map(str, x_tensor.shape)),
            " + ".join(f"[{','.join(map(str, f.shape))}]" for f in feats),
        )
        return feats

    def forward_features(
        self,
        x: Tensor | Image.Image | list[Image.Image],
        make_2D: bool = True,
    ) -> Tensor:
        """Extract

        Args:
            x (Tensor | Image.Image | list[Image.Image]): input image
            make_2D (bool, optional): make features 2D, based on size inferred from `x`, `stride`, and `patch_size`. Defaults to True.

        Raises:
            TypeError: if ViT backbone isn't parsed by protocol

        Returns:
            Tensor: [B,C,N] or [B,C,H,W] (if `make_2D=True`) features for `x`
        """
        x_tensor = self.preprocess_input(x)
        b, _, _, _ = x_tensor.shape
        n_patch_h, n_patch_w = self._get_tokens_per_dim(x_tensor)

        if satisfies_protocol(self.vit, TimmStyleViTGetIntermediates):
            feats = self.vit.forward_features(x_tensor)
            feats = feats[:, self.num_prefix_tokens :]
        elif satisfies_protocol(self.vit, Dv3StyleViT):
            # dv3 returns dict; x_norm_patchtokens already strips prefix tokens
            feats = self.vit.forward_features(x_tensor)["x_norm_patchtokens"]
        else:
            raise TypeError(f"Unsupported ViT type: {type(self.vit)}")
        feats = feats.permute(0, 2, 1)

        if make_2D:
            feats = feats.view(b, self.embed_dim, n_patch_h, n_patch_w)

        logger.info(
            "Forward Features: x: [%s] -> f: [%s]",
            ",".join(map(str, x_tensor.shape)),
            ",".join(map(str, feats.shape)),
        )
        return feats
