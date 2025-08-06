import torch
from PIL import Image
import cv2
import numpy as np
import os
from torchvision import transforms

# from labelling_app import LabellingApp, Transform
from training.arch import AutoSpriteTransformModel  # your model definition
from dataset_generation.labelling_app import Transform, LabellingApp

# --- Step 1: Load the model ---
device = torch.device("cpu") # test on cpu first

model = AutoSpriteTransformModel()
model.load_state_dict(torch.load(input("Enter the path to the model file: "), map_location=device))
model.to(device)
model.eval()

# --- Step 2: Load and preprocess the input image ---
image_path = input("Enter the path to the image file: ")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file {image_path} does not exist.")
img = Image.open(image_path).convert("RGBA")
labeller = LabellingApp()

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # match training size
    transforms.ToTensor(),          # convert to [0,1]
])

input_tensor = transform(img).unsqueeze(0).to(device)  # shape: (1, C, H, W)

# --- Step 3: Inference ---
with torch.no_grad():
    rot_pred, scale_pred = model(input_tensor)

# --- Step 4: Interpret output ---
rotation = rot_pred.item()         # in radians
scale = scale_pred.item()

# print(f"Flip: {is_flipped} (p={flip_prob:.3f})")
print(f"Rotation: {rotation:.3f} rad ({torch.rad2deg(rot_pred).item():.1f}°)")
print(f"Scale: {scale:.3f}")
in_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
out_img = labeller.apply_transform(in_img, Transform(flip=False, rot=rotation, scale=scale))
cv2.imshow("Transformed Image", out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

