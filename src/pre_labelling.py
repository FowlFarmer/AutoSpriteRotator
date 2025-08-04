import csv
import numpy as np
import os
import cv2

from image_generation import ImageGenerator
from remove_background import BackgroundRemover

input_dataset = os.path.join(os.path.dirname(__file__), "../datasets", "stage_1_1_dataset.csv")
output_dir = os.path.join(os.path.dirname(__file__), "../image_bank", "gen")  # Directory to save output images
stripped_dir = os.path.join(os.path.dirname(__file__), "../image_bank", "stripped")  # Directory to save stripped images

def generate_dataset():
    # Initialize models
    pregenerated_filenames = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    print(pregenerated_filenames)
    bg_remover = BackgroundRemover()
    img_generator = ImageGenerator()
    with open(input_dataset, 'r') as csvfile: # To optimize vram, do all imagen then do all rmbg
        reader = csv.reader(csvfile)
        header = next(reader, None)  # Skip header row
        for row in reader:  # Skip header row
            name = row[0]
            prompt = row[2]
            print(f"Generating images for item {name} prompt: {prompt}")

            # Generate images
            for i in range(5):
                output_name = f"{name}_{str(i)}.png"
                output_path = os.path.join(output_dir, output_name)
                if output_name in pregenerated_filenames:
                    print(f"Image {output_name} already exists, skipping generation.")
                    continue
                else:
                    print(f"Generating image: {output_name}")
                    generated_image = img_generator.generate_image(prompt)
                    cv2.imwrite(output_path, generated_image)
                    print(f"Image saved as: {output_path}")

    filenames = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    for i in range(len(filenames)):
        # Load the generated image
        image_path = os.path.join(output_dir, filenames[i])
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            continue

        # I can't be bothered to add checks for already stripped images
        # since this model is so fast

        # Remove background
        pil_image = bg_remover.remove_background(image)

        # Convert back to OpenCV format
        output_image = bg_remover.pil_to_cv2(pil_image)

        # Set all pixels with alpha 0 to black for a clean tensor
        # mask = output_image[:, :, 3] == 0  # Get mask of pixels with alpha 0
        # then Set those pixels to black (BGR)
        output_image = output_image[output_image[:, :, 3] == 0, :3]  

        # Save the processed image
        output_name = os.path.join(stripped_dir, f"stripped_{filenames[i]}") # already has .png
        cv2.imwrite(output_name, output_image)
        print(f"Processed image saved as: {output_name}")

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(stripped_dir):
        os.makedirs(stripped_dir)
    
    generate_dataset()
    print("Dataset generation completed.")
