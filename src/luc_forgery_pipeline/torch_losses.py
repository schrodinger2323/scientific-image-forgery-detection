from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    return torch.mean((2.0 * intersection + eps) / (denominator + eps))


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs + targets, dim=dims) - intersection
    return torch.mean((intersection + eps) / (union + eps))


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice_loss = 1.0 - dice_from_logits(logits, targets)
    return 0.5 * bce + 0.5 * dice_loss


def primary_logits(outputs) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["mask_logits"]
    return outputs


def edge_targets_from_masks(targets: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=targets.dtype,
        device=targets.device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=targets.dtype,
        device=targets.device,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(targets, kernel_x, padding=1)
    grad_y = F.conv2d(targets, kernel_y, padding=1)
    edges = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    return (edges > 0).float()


def forensic_loss(outputs, targets: torch.Tensor, image_labels: torch.Tensor | None = None) -> torch.Tensor:
    logits = primary_logits(outputs)
    loss = bce_dice_loss(logits, targets)
    if not isinstance(outputs, dict):
        return loss

    edge_logits = outputs.get("edge_logits") or []
    if edge_logits:
        edge_targets = edge_targets_from_masks(targets)
        edge_loss = torch.stack(
            [F.binary_cross_entropy_with_logits(edge_logit, edge_targets) for edge_logit in edge_logits]
        ).mean()
        loss = loss + 0.15 * edge_loss

    aux_logits = outputs.get("aux_logits") or []
    if aux_logits:
        aux_loss = torch.stack([bce_dice_loss(aux_logit, targets) for aux_logit in aux_logits]).mean()
        loss = loss + 0.20 * aux_loss

    if image_labels is not None and outputs.get("image_logits") is not None:
        image_targets = image_labels.float().view(-1, 1)
        image_loss = F.binary_cross_entropy_with_logits(outputs["image_logits"], image_targets)
        loss = loss + 0.10 * image_loss
    return loss
