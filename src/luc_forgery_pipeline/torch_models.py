from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import FORENSIC_MODEL_NAMES, PYTORCH_MODEL_NAMES


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class FixedHighPass(nn.Module):
    """Small fixed forensic residual bank inspired by SRM/noise-view preprocessing."""

    def __init__(self) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
            ],
            dtype=torch.float32,
        )
        weight = kernels[:, None, :, :].repeat(1, 3, 1, 1) / 3.0
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=1)


class ConstrainedConv2d(nn.Conv2d):
    """Bayar-style constrained convolution used as a learnable residual extractor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        center_h = weight.shape[2] // 2
        center_w = weight.shape[3] // 2
        mask = torch.ones_like(weight)
        mask[:, :, center_h, center_w] = 0.0
        constrained = weight * mask
        constrained = constrained / (constrained.sum(dim=(1, 2, 3), keepdim=True) + 1e-6)
        constrained = constrained - mask
        return F.conv2d(x, constrained, self.bias, self.stride, self.padding, self.dilation, self.groups)


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int, channels=(32, 64, 128, 256)) -> None:
        super().__init__()
        blocks = []
        prev = in_channels
        for idx, ch in enumerate(channels):
            stride = 1 if idx == 0 else 2
            blocks.append(nn.Sequential(conv_bn_relu(prev, ch, stride=stride), conv_bn_relu(ch, ch)))
            prev = ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class FPNDecoder(nn.Module):
    def __init__(self, channels: list[int], out_channels: int = 96) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(ch, out_channels, 1) for ch in channels])
        self.smooth = nn.ModuleList([conv_bn_relu(out_channels, out_channels) for _ in channels])
        self.head = nn.Sequential(conv_bn_relu(out_channels, out_channels), nn.Conv2d(out_channels, 1, 1))

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        x = self.lateral[-1](features[-1])
        x = self.smooth[-1](x)
        for i in range(len(features) - 2, -1, -1):
            x = F.interpolate(x, size=features[i].shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.lateral[i](features[i])
            x = self.smooth[i](x)
        return F.interpolate(self.head(x), size=output_size, mode="bilinear", align_corners=False)


class ManTraNetLite(nn.Module):
    """Manipulation-trace baseline with fixed and learnable residual feature branches."""

    def __init__(self) -> None:
        super().__init__()
        self.high_pass = FixedHighPass()
        self.constrained = ConstrainedConv2d(3, 8, kernel_size=5, padding=2, bias=False)
        self.rgb_stem = nn.Sequential(conv_bn_relu(3, 24), conv_bn_relu(24, 24))
        self.trace_stem = nn.Sequential(conv_bn_relu(12, 32), conv_bn_relu(32, 32))
        self.encoder = TinyEncoder(56, channels=(48, 96, 160, 256))
        self.decoder = FPNDecoder([48, 96, 160, 256], out_channels=96)
        self.image_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        residual = torch.cat([self.high_pass(x), self.constrained(x)], dim=1)
        stem = torch.cat([self.rgb_stem(x), self.trace_stem(residual)], dim=1)
        features = self.encoder(stem)
        mask_logits = self.decoder(features, x.shape[-2:])
        image_logits = self.image_head(F.adaptive_avg_pool2d(features[-1], 1).flatten(1))
        return {"mask_logits": mask_logits, "image_logits": image_logits}


class MVSSLikeNet(nn.Module):
    """Multi-view, multi-scale supervised forgery localization baseline."""

    def __init__(self, plus_plus: bool = False) -> None:
        super().__init__()
        self.plus_plus = plus_plus
        self.high_pass = FixedHighPass()
        self.semantic = TinyEncoder(3, channels=(48, 96, 192, 320))
        self.noise = TinyEncoder(4, channels=(32, 64, 128, 192))
        fused_channels = [80, 160, 320, 512]
        self.fuse = nn.ModuleList([conv_bn_relu(ch, ch) for ch in fused_channels])
        self.decoder = FPNDecoder(fused_channels, out_channels=128 if plus_plus else 96)
        self.edge_heads = nn.ModuleList([nn.Conv2d(ch, 1, 1) for ch in fused_channels])
        self.image_head = nn.Linear(fused_channels[-1], 1)
        if plus_plus:
            self.attention = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(ch, max(ch // 4, 16), 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(max(ch // 4, 16), ch, 1),
                        nn.Sigmoid(),
                    )
                    for ch in fused_channels
                ]
            )
            self.aux_heads = nn.ModuleList([nn.Conv2d(ch, 1, 1) for ch in fused_channels[:-1]])
        else:
            self.attention = None
            self.aux_heads = nn.ModuleList()

    def forward(self, x: torch.Tensor):
        semantic_features = self.semantic(x)
        noise_features = self.noise(self.high_pass(x))
        fused = []
        for idx, (sem, noise) in enumerate(zip(semantic_features, noise_features)):
            feat = torch.cat([sem, noise], dim=1)
            feat = self.fuse[idx](feat)
            if self.plus_plus and self.attention is not None:
                feat = feat * self.attention[idx](feat)
            fused.append(feat)
        mask_logits = self.decoder(fused, x.shape[-2:])
        edge_logits = [
            F.interpolate(head(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)
            for head, feat in zip(self.edge_heads, fused)
        ]
        aux_logits = [
            F.interpolate(head(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)
            for head, feat in zip(self.aux_heads, fused[:-1])
        ]
        image_logits = self.image_head(F.adaptive_avg_pool2d(fused[-1], 1).flatten(1))
        return {
            "mask_logits": mask_logits,
            "edge_logits": edge_logits,
            "aux_logits": aux_logits,
            "image_logits": image_logits,
        }


class SelfCorrelationMap(nn.Module):
    def __init__(self, top_k: int = 5, max_positions: int = 1024) -> None:
        super().__init__()
        self.top_k = top_k
        self.max_positions = max_positions

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = features.shape
        if h * w > self.max_positions:
            scale = math.sqrt(self.max_positions / float(h * w))
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            features = F.interpolate(features, size=(new_h, new_w), mode="bilinear", align_corners=False)
            h, w = features.shape[-2:]
        flat = F.normalize(features.flatten(2).transpose(1, 2), dim=-1)
        affinity = torch.bmm(flat, flat.transpose(1, 2))
        eye = torch.eye(affinity.shape[-1], device=features.device, dtype=torch.bool).unsqueeze(0)
        affinity = affinity.masked_fill(eye, -1.0)
        k = min(self.top_k, affinity.shape[-1])
        top_values = affinity.topk(k=k, dim=-1).values.mean(dim=-1)
        corr = top_values.view(b, 1, h, w)
        return corr, affinity


class BusterNetLite(nn.Module):
    """Two-branch copy-move baseline: artifact localization + visual similarity branch."""

    def __init__(self) -> None:
        super().__init__()
        self.artifact = TinyEncoder(3, channels=(48, 96, 160, 256))
        self.similarity = TinyEncoder(3, channels=(48, 96, 160, 256))
        self.correlation = SelfCorrelationMap(top_k=7)
        channels = [96, 192, 320, 512]
        self.fuse = nn.ModuleList([conv_bn_relu(ch + 1, ch // 2) for ch in channels])
        self.decoder = FPNDecoder([48, 96, 160, 256], out_channels=96)
        self.image_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        artifact_features = self.artifact(x)
        similarity_features = self.similarity(x)
        corr, _ = self.correlation(similarity_features[-1])
        fused = []
        for art, sim, fuse in zip(artifact_features, similarity_features, self.fuse):
            corr_i = F.interpolate(corr, size=art.shape[-2:], mode="bilinear", align_corners=False)
            fused.append(fuse(torch.cat([art, sim, corr_i], dim=1)))
        mask_logits = self.decoder(fused, x.shape[-2:])
        image_logits = self.image_head(F.adaptive_avg_pool2d(fused[-1], 1).flatten(1))
        return {"mask_logits": mask_logits, "image_logits": image_logits}


class CMFDFormerLite(nn.Module):
    """Transformer-style copy-move baseline with patch tokens and self-correlation map."""

    def __init__(self, embed_dim: int = 160, depth: int = 4, num_heads: int = 5) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.correlation = SelfCorrelationMap(top_k=7)
        self.local = nn.Sequential(conv_bn_relu(3, 32), conv_bn_relu(32, 64, stride=2), conv_bn_relu(64, 96, stride=2))
        self.decoder = nn.Sequential(
            conv_bn_relu(embed_dim + 1 + 96, 192),
            conv_bn_relu(192, 128),
            nn.Conv2d(128, 1, 1),
        )
        self.image_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor):
        tokens_2d = self.patch(x)
        b, c, h, w = tokens_2d.shape
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        features = tokens.transpose(1, 2).reshape(b, c, h, w)
        corr, _ = self.correlation(features)
        local = F.interpolate(self.local(x), size=features.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([features, corr, local], dim=1))
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        image_logits = self.image_head(tokens.mean(dim=1))
        return {"mask_logits": logits, "image_logits": image_logits}


class DualOrderAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = conv_bn_relu(channels + 2, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        corr, affinity = SelfCorrelationMap(top_k=7)(features)
        first = torch.sigmoid(corr)
        second = torch.softmax(affinity, dim=-1).pow(2).sum(dim=-1)
        second = second.view(features.shape[0], 1, features.shape[2], features.shape[3])
        attended = features * (1.0 + first) * (1.0 + second)
        return self.proj(torch.cat([attended, first, second], dim=1))


class DOAGANLite(nn.Module):
    """Dual-order attention generator baseline for copy-move localization."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyEncoder(3, channels=(48, 96, 160, 256))
        self.dual_attention = DualOrderAttention(256)
        self.decoder = FPNDecoder([48, 96, 160, 256], out_channels=96)
        self.edge_head = nn.Conv2d(96, 1, 1)
        self.image_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        features[-1] = self.dual_attention(features[-1])
        mask_logits = self.decoder(features, x.shape[-2:])
        low_edge = self.edge_head(F.interpolate(features[1], size=x.shape[-2:], mode="bilinear", align_corners=False))
        image_logits = self.image_head(F.adaptive_avg_pool2d(features[-1], 1).flatten(1))
        return {"mask_logits": mask_logits, "edge_logits": [low_edge], "image_logits": image_logits}


class QualityEnhancer(nn.Module):
    """Lightweight residual enhancement front-end inspired by QDL-CMFD's quality module."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_relu(3, 32),
            conv_bn_relu(32, 32),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.15 * self.net(x)


class QDLCMFDLite(nn.Module):
    """Quality-enhanced dual-branch CMFD baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.enhancer = QualityEnhancer()
        self.high_pass = FixedHighPass()
        self.manipulation = TinyEncoder(3, channels=(48, 96, 160, 256))
        self.similarity = TinyEncoder(4, channels=(32, 64, 128, 192))
        self.correlation = SelfCorrelationMap(top_k=7)
        fused_channels = [80, 160, 288, 448]
        self.fuse = nn.ModuleList([conv_bn_relu(ch + 1, out) for ch, out in zip(fused_channels, [64, 128, 192, 256])])
        self.decoder = FPNDecoder([64, 128, 192, 256], out_channels=96)
        self.source_head = nn.Conv2d(128, 1, 1)
        self.image_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        enhanced = self.enhancer(x)
        manipulation_features = self.manipulation(enhanced)
        similarity_features = self.similarity(self.high_pass(enhanced))
        corr, _ = self.correlation(similarity_features[-1])
        fused = []
        for manip, sim, fuse in zip(manipulation_features, similarity_features, self.fuse):
            corr_i = F.interpolate(corr, size=manip.shape[-2:], mode="bilinear", align_corners=False)
            fused.append(fuse(torch.cat([manip, sim, corr_i], dim=1)))
        mask_logits = self.decoder(fused, x.shape[-2:])
        source_logits = self.source_head(F.interpolate(fused[1], size=x.shape[-2:], mode="bilinear", align_corners=False))
        image_logits = self.image_head(F.adaptive_avg_pool2d(fused[-1], 1).flatten(1))
        return {"mask_logits": mask_logits, "aux_logits": [source_logits], "image_logits": image_logits}


class SiameseCMFDLite(nn.Module):
    """Shared-weight dual-branch network for copy-move similarity localization."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = TinyEncoder(3, channels=(48, 96, 160, 256))
        self.correlation = SelfCorrelationMap(top_k=9, max_positions=1024)
        self.fuse = nn.ModuleList([conv_bn_relu(ch * 3 + 1, ch) for ch in [48, 96, 160, 256]])
        self.decoder = FPNDecoder([48, 96, 160, 256], out_channels=96)
        self.image_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        blurred = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        a_features = self.shared(x)
        b_features = self.shared(blurred)
        corr, _ = self.correlation(a_features[-1])
        fused = []
        for a, b, fuse in zip(a_features, b_features, self.fuse):
            corr_i = F.interpolate(corr, size=a.shape[-2:], mode="bilinear", align_corners=False)
            fused.append(fuse(torch.cat([a, b, torch.abs(a - b), corr_i], dim=1)))
        mask_logits = self.decoder(fused, x.shape[-2:])
        image_logits = self.image_head(F.adaptive_avg_pool2d(fused[-1], 1).flatten(1))
        return {"mask_logits": mask_logits, "image_logits": image_logits}


class SelfCorrelationCMFDLite(nn.Module):
    """Similarity-map CMFD baseline built around multi-scale self-correlation cues."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyEncoder(3, channels=(48, 96, 160, 256))
        self.correlation = SelfCorrelationMap(top_k=9)
        self.enrich = nn.ModuleList([conv_bn_relu(ch + 1, ch) for ch in [48, 96, 160, 256]])
        self.decoder = FPNDecoder([48, 96, 160, 256], out_channels=96)
        self.edge_heads = nn.ModuleList([nn.Conv2d(ch, 1, 1) for ch in [48, 96, 160, 256]])
        self.image_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        corr_maps = []
        enriched = []
        for feat, enrich in zip(features, self.enrich):
            pooled = feat
            while pooled.shape[-2] * pooled.shape[-1] > 1024:
                pooled = F.avg_pool2d(pooled, kernel_size=2, stride=2, ceil_mode=True)
            corr, _ = self.correlation(pooled)
            corr = F.interpolate(corr, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            corr_maps.append(corr)
            enriched.append(enrich(torch.cat([feat, corr], dim=1)))
        mask_logits = self.decoder(enriched, x.shape[-2:])
        edge_logits = [
            F.interpolate(head(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)
            for head, feat in zip(self.edge_heads, enriched)
        ]
        image_logits = self.image_head(F.adaptive_avg_pool2d(enriched[-1], 1).flatten(1))
        return {"mask_logits": mask_logits, "edge_logits": edge_logits, "image_logits": image_logits}


class SegFormerB0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:
            print(f"SegFormer-B0 pretrained weights unavailable ({exc}); using random MIT-B0 initialization.")
            config = SegformerConfig(num_labels=1)
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class DeepLabV3Plus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The pilot stability experiment must run reproducibly in Colab without
        # hanging on optional package imports or pretrained-weight downloads.
        # Set USE_EXTERNAL_DEEPLAB=1 only when segmentation_models_pytorch is
        # installed and you explicitly want that backend.
        import os

        if os.environ.get("USE_EXTERNAL_DEEPLAB", "0") != "1":
            print("Using internal lightweight DeepLabV3+ implementation. Set USE_EXTERNAL_DEEPLAB=1 for SMP backend.")
            self.model = InternalDeepLabV3Plus()
            self.backend = "internal_lightweight_deeplabv3plus"
            return
        try:
            import segmentation_models_pytorch as smp

            self.model = smp.DeepLabV3Plus(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation=None,
            )
            self.backend = "smp_deeplabv3plus"
        except Exception as exc:
            print(f"DeepLabV3+ via segmentation_models_pytorch unavailable ({exc}).")
            try:
                from torchvision.models.segmentation import deeplabv3_resnet50

                print("Falling back to torchvision deeplabv3_resnet50 with a 1-channel classifier head.")
                self.model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1)
                self.backend = "torchvision_deeplabv3_resnet50_fallback"
            except Exception as inner_exc:
                print(f"torchvision DeepLab fallback unavailable ({inner_exc}).")
                print("Falling back to internal lightweight DeepLabV3+ implementation.")
                self.model = InternalDeepLabV3Plus()
                self.backend = "internal_lightweight_deeplabv3plus"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out["out"]
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 128, rates=(1, 6, 12, 18)) -> None:
        super().__init__()
        branches = []
        for rate in rates:
            if rate == 1:
                branches.append(conv_bn_relu(in_channels, out_channels, kernel_size=1))
            else:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.project = conv_bn_relu(out_channels * (len(rates) + 1), out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        pooled = F.interpolate(self.image_pool(x), size=size, mode="bilinear", align_corners=False)
        features.append(pooled)
        return self.project(torch.cat(features, dim=1))


class InternalDeepLabV3Plus(nn.Module):
    """Dependency-free DeepLabV3+ style fallback for Colab environments without SMP."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyEncoder(3, channels=(48, 96, 192, 320))
        self.aspp = ASPP(320, 128)
        self.low_project = conv_bn_relu(48, 48, kernel_size=1)
        self.decoder = nn.Sequential(
            conv_bn_relu(176, 128),
            conv_bn_relu(128, 128),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        low = self.low_project(features[0])
        high = self.aspp(features[-1])
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([low, high], dim=1))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class DinoV2Segmentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model

        try:
            self.encoder = Dinov2Model.from_pretrained("facebook/dinov2-small")
        except Exception as exc:
            print(f"DINOv2-small pretrained weights unavailable ({exc}); using random DINOv2-small initialization.")
            self.encoder = Dinov2Model(Dinov2Config())
        hidden = int(self.encoder.config.hidden_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=x, interpolate_pos_encoding=True)
        tokens = outputs.last_hidden_state[:, 1:, :]
        grid = int(math.sqrt(tokens.shape[1]))
        tokens = tokens[:, : grid * grid, :]
        features = tokens.transpose(1, 2).reshape(x.shape[0], -1, grid, grid)
        logits = self.decoder(features)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_torch_model(model_name: str) -> nn.Module:
    if model_name == "segformer_b0":
        return SegFormerB0()
    if model_name == "deeplabv3plus":
        return DeepLabV3Plus()
    if model_name == "dinov2_seg":
        return DinoV2Segmentation()
    if model_name == "mantranet":
        return ManTraNetLite()
    if model_name == "mvssnet":
        return MVSSLikeNet(plus_plus=False)
    if model_name == "mvssnetpp":
        return MVSSLikeNet(plus_plus=True)
    if model_name == "busternet":
        return BusterNetLite()
    if model_name == "cmfdformer":
        return CMFDFormerLite()
    if model_name == "doagan":
        return DOAGANLite()
    if model_name == "qdl_cmfd":
        return QDLCMFDLite()
    if model_name == "siamese_cmfd":
        return SiameseCMFDLite()
    if model_name == "selfcorr_cmfd":
        return SelfCorrelationCMFDLite()
    available = PYTORCH_MODEL_NAMES + FORENSIC_MODEL_NAMES
    raise ValueError(f"Unknown PyTorch model '{model_name}'. Available: {', '.join(available)}")
