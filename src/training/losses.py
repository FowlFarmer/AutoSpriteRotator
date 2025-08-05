import torch
import torch.nn.functional as F


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
    Rotation loss conditionally flips the true_rot based on predicted flip.
    Scale loss is standard MSE.

    Returns a dict of all components and total loss.
    """
    # Flip prediction: sigmoid for probability, then binary decision
    flip_prob = torch.sigmoid(flip_logit)
    flip_pred = (flip_prob > 0.5).bool()

    # Flip loss (before sigmoid)
    loss_flip = F.binary_cross_entropy_with_logits(flip_logit, true_flip)

    # Rotation target depends on predicted flip
    flip_mismatch = flip_pred != true_flip.bool()
    rot_target = torch.where(flip_mismatch, -true_rot, true_rot)
    loss_rot = F.mse_loss(rot_pred, rot_target)

    # Scale loss
    loss_scale = F.mse_loss(scale_pred, true_scale)

    # Total loss (you can weight these if needed)
    total_loss = loss_flip + 4 * loss_rot + 4 * loss_scale

    return {
        "loss_total": total_loss,
        "loss_flip": loss_flip,
        "loss_rot": loss_rot,
        "loss_scale": loss_scale
    }
