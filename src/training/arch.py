import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoSpriteTransformModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Normalization parameters for 4-channel input (RGBA)
        self.mean = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)

        # Feature extractor (e.g., from 1024x1024 image)
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        
        self.flatten = nn.Flatten()  # [B, 64, 1, 1] → [B, 64]
        self.fc_shared = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU()
        )

        # Flip head (1 output, sigmoid applied later)
        self.flip_head = nn.Linear(256, 1)

        # Rotation head (takes shared + flip_pred as input)
        self.rot_head = nn.Sequential(
            nn.Linear(256 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Rotation angle (float)
        )

        # Scale head (independent)
        self.scale_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, x):
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        x = (x - self.mean) / self.std
        x = self.backbone(x)            # [B, 128, 4, 4]
        x = self.flatten(x)             # [B, 2048]
        shared_feat = self.fc_shared(x) # [B, 256]

        flip_logit = self.flip_head(shared_feat)  # [B, 1]
        flip_prob = torch.sigmoid(flip_logit)     # [B, 1]

        # Optional: stop gradients from flip flowing into rot (your choice)
        flip_detached = flip_prob.detach()

        # Concatenate flip into rotation input
        rot_input = torch.cat([shared_feat, flip_detached], dim=1)
        rotation = self.rot_head(rot_input)

        scale = self.scale_head(shared_feat)

        return flip_logit, rotation, scale
