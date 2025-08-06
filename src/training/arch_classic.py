import torch
import torch.nn as nn

class AutoSpriteTransformModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Normalization parameters for 4-channel input (RGBA)
        self.mean = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)

        # Feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # downsample 1
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # downsample 2
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))  # preserves spatial grid
        )

        self.flatten = nn.Flatten()
        self.fc_shared = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.rot_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.scale_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        x = (x - self.mean) / self.std
        x = self.backbone(x)
        x = self.flatten(x)             # [B, 256]
        shared_feat = self.fc_shared(x) # [B, 512]

        rotation = self.rot_head(shared_feat)
        scale = self.scale_head(shared_feat)
        return rotation, scale