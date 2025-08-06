import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os
import sys

from dataset import TransformLabelDataset
from arch import AutoSpriteTransformModel
from losses import compute_flip_rot_scale_loss, compute_rot_scale_loss

def train(model, dataset, checkpoint_path, ckpt_name="model", load_ckpt_file=None, epochs=10, batch_size=16, lr=1e-4, device="cuda"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    start_epoch = 0
    if load_ckpt_file is not None:
        # resume from next epoch, discard optimizer state switching from training to tuning
        print(f"Loading checkpoint from {load_ckpt_file}")
        model.load_state_dict(torch.load(load_ckpt_file, map_location=device))

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        # total_flip_loss = 0.0
        total_rot_loss = 0.0
        total_scale_loss = 0.0
        few_batches_loss_sliding_window = []
        # few_batches_flip_loss_sliding_window = []
        few_batches_rot_loss_sliding_window = []
        few_batches_scale_loss_sliding_window = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, true_rot, true_scale = [b.to(device) for b in batch]

            rot_pred, scale_pred = model(images)
        
            loss_dict = compute_rot_scale_loss(
                rot_pred, scale_pred,
                true_rot, true_scale
            )

            loss = loss_dict["loss_total"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            few_batches_loss_sliding_window.append(loss.item())
            # few_batches_flip_loss_sliding_window.append(loss_dict["loss_flip"].item())
            few_batches_rot_loss_sliding_window.append(loss_dict["loss_rot"].item())
            few_batches_scale_loss_sliding_window.append(loss_dict["loss_scale"].item())

            total_loss += loss.item()
            # total_flip_loss += loss_dict["loss_flip"].item()
            total_rot_loss += loss_dict["loss_rot"].item()
            total_scale_loss += loss_dict["loss_scale"].item()
            avg_few_batch_loss = sum(few_batches_loss_sliding_window) / len(few_batches_loss_sliding_window)
            # avg_few_batch_flip_loss = sum(few_batches_flip_loss_sliding_window) / len(few_batches_flip_loss_sliding_window)
            avg_few_batch_rot_loss = sum(few_batches_rot_loss_sliding_window) / len(few_batches_rot_loss_sliding_window)
            avg_few_batch_scale_loss = sum(few_batches_scale_loss_sliding_window) / len(few_batches_scale_loss_sliding_window)
            if len(few_batches_loss_sliding_window) >= 5:
                print(f"Batch Loss Avg (Last 5 Batches): {avg_few_batch_loss:.4f}, "
                    #   f"Flip Loss Avg: {avg_few_batch_flip_loss:.4f}, "
                      f"Rot Loss Avg: {avg_few_batch_rot_loss:.4f}, "
                      f"Scale Loss Avg: {avg_few_batch_scale_loss:.4f}")
                few_batches_loss_sliding_window.clear()
                # few_batches_flip_loss_sliding_window.clear()
                few_batches_rot_loss_sliding_window.clear()
                few_batches_scale_loss_sliding_window.clear()

        avg_loss = total_loss / len(dataloader)
        # avg_flip_loss = total_flip_loss / len(dataloader)
        avg_rot_loss = total_rot_loss / len(dataloader)
        avg_scale_loss = total_scale_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Rot Loss: {avg_rot_loss:.4f}, Avg Scale Loss: {avg_scale_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f"{ckpt_name}_epoch_{epoch+1}.pt"))

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
        train(model, dataset, os.path.join(os.path.dirname(__file__), "../../checkpoints"), ckpt_name="tune_noflip", load_ckpt_file=load_checkpoint, epochs=25, batch_size=32, lr=1e-5, device="cuda")
    else:
        train(model, dataset, os.path.join(os.path.dirname(__file__), "../../checkpoints"), ckpt_name="train_noflip", load_ckpt_file=None, epochs=10, batch_size=32, lr=1e-4, device="cuda")
