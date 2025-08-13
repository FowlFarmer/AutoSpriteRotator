import torch
import torch.nn.functional as F
from torch.special import i0  # Bessel function I₀
# import math
import matplotlib.pyplot as plt

def angle_difference(a, b):
    """Returns (a - b) wrapped to [-π, π]."""
    return (a - b + torch.pi) % (2 * torch.pi) - torch.pi

def compute_accuracy_within_threshold(pred, target, threshold_deg):
    angle_diff = angle_difference(pred, target)  # radians
    error_deg = torch.abs(angle_diff) * 180 / torch.pi  # convert to degrees

    correct = (error_deg <= threshold_deg).float()
    return correct.mean()  # scalar accuracy

def von_mises_nll(
    mu: torch.Tensor, 
    kappa_raw: torch.Tensor, 
    scale_pred: torch.Tensor, 
    true_rot: torch.Tensor, 
    true_scale: torch.Tensor
) -> dict:
    # Ensure kappa > 0 (concentration)
    kappa = torch.clamp(F.softplus(kappa_raw), min=0.1)

    # Angle difference wrapped
    angle_diff = angle_difference(true_rot, mu)

    # NLL = -κ cos(θ - μ) + log(2π I₀(κ))
    von_mises = torch.mean(-kappa * torch.cos(angle_diff) + torch.log(2 * math.pi * i0(kappa)))

    # Scale loss (MSE)
    scale_loss = F.mse_loss(scale_pred, true_scale)

    # Total loss
    total_loss = von_mises + scale_loss
    return {
        "loss_total": total_loss,
        "loss_rot": von_mises,
        "loss_scale": scale_loss,
        "kappa": kappa.mean(),
        "mu": angle_diff.mean()  # Mean angle difference for logging
    }

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

def this_somehow_works(
    # flip_logit: torch.Tensor,     # shape [B, 1]
    rot_pred: torch.Tensor,       # shape [B, 1]
    scale_pred: torch.Tensor,     # shape [B, 1]
    # true_flip: torch.Tensor,      # shape [B, 1], values 0.0 or 1.0
    true_rot: torch.Tensor,       # shape [B, 1], radians
    true_scale: torch.Tensor      # shape [B, 1], float
) -> dict:

    # Rotation loss (angle-aware + Gaussian bump relative to label)

    # Data-dependent penalty centers: ±π/2 from the true target
    penalty_strength = 300.0
    penalty_width = 1.883  # adjust as needed

    # Penalty bumps relative to ground truth
    bump_pos = torch.exp(-((rot_pred - (true_rot + math.pi / 2)) ** 2) / (2 * penalty_width ** 2))
    bump_neg = torch.exp(-((rot_pred - (true_rot - math.pi / 2)) ** 2) / (2 * penalty_width ** 2))
    penalty_mask = bump_pos + bump_neg

    # Final weighted rotation loss
    angle_loss = (rot_pred - true_rot)**2  # wraparound-safe diff
    weighted_rot_loss = 15 * penalty_strength * penalty_mask + (angle_loss/10)
    loss_rot = torch.mean(weighted_rot_loss)
    mask = torch.mean(15*penalty_strength * penalty_mask)
    angle = torch.mean(angle_loss/10)

    # Scale loss
    loss_scale = 25 * F.mse_loss(scale_pred, true_scale)

    # Total loss (feel free to tune these weights)
    total_loss = loss_rot + loss_scale

    return {
        "loss_total": total_loss,
        # "loss_flip": loss_flip,
        "loss_rot": loss_rot,
        "loss_scale": loss_scale,
        "penalty_mask": mask,  # for debugging
        "angle_diff": angle  # for debugging
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

    # Data-dependent penalty centers: ±π/2 from the true target
    # --- Hyperparameters ---
    penalty_strength = 500.0
    penalty_width = 1.283  # Controls bump falloff calculated to 5 deg at correct angle

    # --- Penalty bumps centered at ±90° from the ground truth ---
    bump_pos = torch.exp(
        -((rot_pred - (true_rot + math.pi / 2)) ** 2) / (2 * penalty_width ** 2)
    )
    bump_neg = torch.exp(
        -((rot_pred - (true_rot - math.pi / 2)) ** 2) / (2 * penalty_width ** 2)
    )
    penalty_mask = bump_pos + bump_neg  # Shape: (batch,)

    # --- Core angle loss using wraparound-safe difference ---
    angle_loss = (angle_difference(rot_pred, true_rot)) ** 2  # Still in radians²

    # --- Weighted final loss ---
    weighted_rot_loss = penalty_strength * penalty_mask + (angle_loss)
    loss_rot = torch.mean(weighted_rot_loss)

    # --- Optional logging terms ---
    mask = torch.mean(penalty_strength * penalty_mask)  # Mean bump magnitude
    angle = torch.mean(angle_loss)                       # Mean core rot loss

    # --- Scale loss ---
    loss_scale = 25 * F.mse_loss(scale_pred, true_scale)


    # Total loss (feel free to tune these weights)
    total_loss = loss_rot + loss_scale

    return {
        "loss_total": total_loss,
        # "loss_flip": loss_flip,
        "loss_rot": loss_rot,
        "loss_scale": loss_scale,
        "penalty_mask": mask,  # for debugging
        "angle_diff": angle  # for debugging
    }

def compute_basic_loss(
    # flip_logit: torch.Tensor,     # shape [B, 1]
    rot_pred: torch.Tensor,       # shape [B, 1]
    scale_pred: torch.Tensor,     # shape [B, 1]
    # true_flip: torch.Tensor,      # shape [B, 1], values 0.0 or 1.0
    true_rot: torch.Tensor,       # shape [B, 1], radians
    true_scale: torch.Tensor      # shape [B, 1], float
) -> dict:

    # --- Scale loss ---
    loss_scale = F.mse_loss(scale_pred, true_scale)

    # --- Rotation loss ---
    angle_diff = angle_difference(rot_pred, true_rot)  # shape [B, 1]
    loss_rot = torch.mean(torch.atan(angle_diff**2))   # arctan(x^2) loss, check readme for explanation

    # Total loss (feel free to tune these weights)
    total_loss = loss_rot + loss_scale

    return {
        "loss_total": total_loss,
        # "loss_flip": loss_flip,
        "loss_rot": loss_rot,
        "loss_scale": loss_scale,
        "mAP15": compute_accuracy_within_threshold(rot_pred, true_rot, 15.0),  # Mean Average Precision at 15 degrees
        "mAP30": compute_accuracy_within_threshold(rot_pred, true_rot, 30.0)  # Mean Average Precision at 30 degrees
    }

def compute_cos_loss(
    # flip_logit: torch.Tensor,     # shape [B, 1]
    rot_pred: torch.Tensor,       # shape [B, 1]
    scale_pred: torch.Tensor,     # shape [B, 1]
    # true_flip: torch.Tensor,      # shape [B, 1], values 0.0 or 1.0
    true_rot: torch.Tensor,       # shape [B, 1], radians
    true_scale: torch.Tensor      # shape [B, 1], float
) -> dict:

    # --- Scale loss ---
    loss_scale = F.mse_loss(scale_pred, true_scale)

    # --- Rotation loss ---
    angle_diff = angle_difference(rot_pred, true_rot)  # shape [B, 1]
    loss_rot = torch.mean(1-torch.cos(angle_diff))   # cos(x) loss, check readme for explanation

    # Total loss (feel free to tune these weights)
    total_loss = loss_rot + loss_scale

    return {
        "loss_total": total_loss,
        # "loss_flip": loss_flip,
        "loss_rot": loss_rot,
        "loss_scale": loss_scale,
        "mAP15": compute_accuracy_within_threshold(rot_pred, true_rot, 15.0),  # Mean Average Precision at 15 degrees
        "mAP30": compute_accuracy_within_threshold(rot_pred, true_rot, 30.0)  # Mean Average Precision at 30 degrees
    }

