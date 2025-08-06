import torch
import torch.nn.functional as F
import math

def angle_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the shortest angular difference between two angles in radians.
    Output is in [-π, π].
    """
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))

def compute_flip_rot_scale_loss(
    flip_logit: torch.Tensor,     # shape [B, 1]
    rot_pred: torch.Tensor,       # shape [B, 1]
    scale_pred: torch.Tensor,     # shape [B, 1]
    true_flip: torch.Tensor,      # shape [B, 1], values 0.0 or 1.0
    true_rot: torch.Tensor,       # shape [B, 1], radians
    true_scale: torch.Tensor      # shape [B, 1], float
) -> dict:
    """
    Computes the combined loss for flip, rotation, and scale.

    Flip loss uses BCEWithLogits.
    Rotation loss conditionally flips the true_rot based on predicted flip,
    and applies a Gaussian penalty around ±π/2.
    Scale loss is standard MSE.

    Returns a dict of all components and total loss.
    """
    # Flip prediction
    flip_prob = torch.sigmoid(flip_logit)
    flip_pred = (flip_prob > 0.5).bool()

    # Flip loss
    loss_flip = F.binary_cross_entropy_with_logits(flip_logit, true_flip)

    # Flip mismatch correction
    flip_mismatch = flip_pred != true_flip.bool()
    rot_target = torch.where(flip_mismatch, -true_rot, true_rot)

    # Rotation loss (angle-aware + Gaussian bump penalty near ±π/2)
    angle_diff = angle_difference(rot_pred, rot_target)

    # Gaussian bump penalty around ±π/2
    penalty_strength = 1000.0
    penalty_width = 0.9

    bump_pos = torch.exp(-((rot_pred - (math.pi / 2)) ** 2) / (2 * penalty_width ** 2))
    bump_neg = torch.exp(-((rot_pred + (math.pi / 2)) ** 2) / (2 * penalty_width ** 2))
    penalty_mask = bump_pos + bump_neg

    weighted_rot_loss = (1 + penalty_strength * penalty_mask) * (angle_diff ** 2)
    loss_rot = torch.mean(weighted_rot_loss)

    # Scale loss
    loss_scale = F.mse_loss(scale_pred, true_scale)

    # Total loss (feel free to tune these weights)
    total_loss = loss_flip + 4 * loss_rot + 4 * loss_scale

    return {
        "loss_total": total_loss,
        "loss_flip": loss_flip,
        "loss_rot": loss_rot,
        "loss_scale": loss_scale
    }

def compute_rot_scale_loss(
    # flip_logit: torch.Tensor,     # shape [B, 1]
    rot_pred: torch.Tensor,       # shape [B, 1]
    scale_pred: torch.Tensor,     # shape [B, 1]
    # true_flip: torch.Tensor,      # shape [B, 1], values 0.0 or 1.0
    true_rot: torch.Tensor,       # shape [B, 1], radians
    true_scale: torch.Tensor      # shape [B, 1], float
) -> dict:

    # Rotation loss (angle-aware + Gaussian bump relative to label)
    angle_diff = angle_difference(rot_pred, true_rot)  # wraparound-safe diff

    # Data-dependent penalty centers: ±π/2 from the true target
    penalty_strength = 50.0
    penalty_width = 0.25  # adjust as needed

    # Penalty bumps relative to ground truth
    bump_pos = torch.exp(-((rot_pred - (true_rot + math.pi / 2)) ** 2) / (2 * penalty_width ** 2))
    bump_neg = torch.exp(-((rot_pred - (true_rot - math.pi / 2)) ** 2) / (2 * penalty_width ** 2))
    penalty_mask = bump_pos + bump_neg

    # Final weighted rotation loss
    weighted_rot_loss = (1 + penalty_strength * penalty_mask) * (angle_diff ** 2)
    loss_rot = torch.mean(weighted_rot_loss)


    # Scale loss
    loss_scale = F.mse_loss(scale_pred, true_scale)

    # Total loss (feel free to tune these weights)
    total_loss = loss_rot + loss_scale

    return {
        "loss_total": total_loss,
        # "loss_flip": loss_flip,
        "loss_rot": loss_rot,
        "loss_scale": loss_scale
    }
