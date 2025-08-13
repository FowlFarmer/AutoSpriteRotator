import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import sys
import math
import csv


from dataset import TransformLabelDataset
from arch_classic import AutoSpriteTransformModel
from losses import compute_flip_rot_scale_loss, compute_rot_scale_loss, this_somehow_works, von_mises_nll, compute_basic_loss, compute_cos_loss
# from visualizer import LiveMetricPlotter

def train(model, dataset, checkpoint_path, ckpt_name="model", load_ckpt_file=None, epochs=10, batch_size=16, lr=1e-4, device="cuda", enable_csv_datalogging = False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
    threshold=1e-4,
    threshold_mode='rel',
    verbose=False
    )
    # Initialize live plotter
    # plotter = LiveMetricPlotter(tracked_keys=["loss_total", "loss_rot", "loss_scale", "mAP15", "mAP30"])
    # not available on ssh

    if enable_csv_datalogging:
        with open(os.path.join(checkpoint_path, f"{ckpt_name}_training_log.csv"), mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            header = ["epoch", "batch", "learning_rate", "loss_total", "loss_rot", "loss_scale", "mAP15", "mAP30"]
            csv_writer.writerow(header)

    start_epoch = 0
    if load_ckpt_file is not None:
        # resume from next epoch, discard optimizer state switching from training to tuning
        print(f"Loading checkpoint from {load_ckpt_file}")
        model.load_state_dict(torch.load(load_ckpt_file, map_location=device))

    batches = 0 # for plotter, logging
    learning_rate = float(scheduler.get_last_lr()[0])  # this returns a list for some reason
    print(f"Initial learning rate: {learning_rate}")

    for epoch in range(start_epoch, epochs):
        model.train()
        # Initialize loss tracking
        loss_totals = {"loss_total": 0.0, "loss_rot": 0.0, "loss_scale": 0.0, "mAP15": 0.0, "mAP30": 0.0}
        sliding_windows = {key: [] for key in loss_totals.keys()}
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, true_rot, true_scale = [b.to(device) for b in batch]

            rotation, scale = model(images)

            loss_dict = compute_cos_loss(
                rotation, scale,
                true_rot, true_scale
            )

            loss = loss_dict["loss_total"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update sliding windows and totals
            batch_losses = {
                "loss_total": loss.item(),
                "loss_rot": loss_dict["loss_rot"].item(),
                "loss_scale": loss_dict["loss_scale"].item(),
                "mAP15": loss_dict["mAP15"].item(),
                "mAP30": loss_dict["mAP30"].item(),
                # "kappa": loss_dict["kappa"].item(),
                # "mu": abs(loss_dict["mu"].item())
            }


            if enable_csv_datalogging:
                with open(os.path.join(checkpoint_path, f"{ckpt_name}_training_log.csv"), mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    row = [epoch+1, batches, learning_rate] + [batch_losses[key] for key in header[3:]]
                    csv_writer.writerow(row)

            # plotter.update(batches, batch_losses)
            batches += 1

            for key, value in batch_losses.items():
                sliding_windows[key].append(value)
                loss_totals[key] += value

            # Print sliding window averages every 5 batches
            if len(sliding_windows["loss_total"]) >= 5:
                avg_losses = {key: sum(window) / len(window) for key, window in sliding_windows.items()}
                keys = [key for key in avg_losses.keys()]
                print(f"Batch Loss Avg (Last 5 Batches): {', '.join(f'{key}: {avg_losses[key]:.4f}' for key in keys)}")
                # Clear all sliding windows
                for window in sliding_windows.values():
                    window.clear()

        # Calculate and print epoch averages
        epoch_averages = {key: total / len(dataloader) for key, total in loss_totals.items()}
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_averages['loss_total']:.4f}, "
              f"Rot: {epoch_averages['loss_rot']:.4f}, Scale: {epoch_averages['loss_scale']:.4f}, mAP15: {epoch_averages['mAP15']:.4f}, mAP30: {epoch_averages['mAP30']:.4f}")
            #   f"kappa diff: {epoch_averages['kappa']:.4f}, mu Diff: {epoch_averages['mu']:.4f}")
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f"{ckpt_name}_epoch_{epoch+1}.pt"))

        # Update learning rate scheduler
        scheduler.step((epoch_averages["mAP30"]+epoch_averages["mAP15"])/2)
        learning_rate = scheduler.get_last_lr()[0] # Get the current learning rate
        print(f"Learning rate: {learning_rate}")  # Get the current learning rate
        # Use avg of mAP15 and mAP30 as metric for scheduler

if __name__ == "__main__":
    import os
    from torchvision import transforms

    # Define paths
    images_dir = os.path.join(os.path.dirname(__file__), "../../image_bank/variants_2")
    label_path = os.path.join(os.path.dirname(__file__), "../../datasets/variants_labels_2.csv")

    # Create dataset
    dataset = TransformLabelDataset(images_dir, label_path)

    # Initialize model
    model = AutoSpriteTransformModel()

    # Start training
    if not torch.cuda.is_available():
        print("CUDA is not available, quitting.")
        sys.exit(1)
    if(input("Train or tune? (tune/train): ").strip().lower() == 'tune'):
        load_checkpoint = input("Enter the path to the checkpoint file: ").strip()
        train(model, dataset, os.path.join(os.path.dirname(__file__), "../../checkpoints"), ckpt_name="v5_cosx", load_ckpt_file=load_checkpoint, epochs=25, batch_size=8, lr=1e-5, device="cuda", enable_csv_datalogging=True)
    else:
        load_checkpoint = input("Enter the path to the checkpoint file (n for skip): ").strip()
        train(model, dataset, os.path.join(os.path.dirname(__file__), "../../checkpoints"), ckpt_name="v5_cosx", load_ckpt_file=load_checkpoint if load_checkpoint != "n" else None, epochs=35, batch_size=8, lr=1e-4, device="cuda", enable_csv_datalogging=True)
