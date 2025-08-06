import torch
from PIL import Image
import cv2
import numpy as np
import os
from torchvision import transforms
import math
# from labelling_app import LabellingApp, Transform
from training.arch_classic import AutoSpriteTransformModel  # your model definition
from dataset_generation.labelling_app import Transform, LabellingApp

# --- Step 1: Load the model ---
device = torch.device("cpu") # test on cpu first

model = AutoSpriteTransformModel()
model.load_state_dict(torch.load(input("Enter the path to the model file: "), map_location=device))
model.to(device)
model.eval()

# --- Step 2: Load and preprocess the input image ---
auto_do: bool = input("Auto transform all? (y/n): ").strip().lower() == 'y'
num = 0
if auto_do:
    dir = input("Enter the directory containing images: ")
    image_files = [f for f in os.listdir(dir) if f.endswith((".png"))]
i = 0
while True:
    if auto_do:
        image_path = os.path.join(dir, image_files[0])
        image_files.pop(0)
    else:
        image_path = input("Enter the image filename: ")
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
    rotation, scale = 0, 0
    with torch.no_grad():
        rotation, scale = model(input_tensor)

    # --- Step 4: Interpret output ---
    rotation = rotation.item()         # in radians
    scale = scale.item()

    # print(f"Flip: {is_flipped} (p={flip_prob:.3f})")
    print(f"Rotation: {rotation:.3f} rad ({rotation * 180 / math.pi:.3f}°)")
    print(f"Scale: {scale:.3f}, num {i}")
    i += 1
    in_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    out_img = labeller.apply_transform(in_img, Transform(flip=False, rot=rotation, scale=scale))
    if auto_do:
        num += 1
        name = f"transformed_{num}.png"
    else:
        name = "transformed_image.png"
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "../image_bank/test2", name), out_img)

