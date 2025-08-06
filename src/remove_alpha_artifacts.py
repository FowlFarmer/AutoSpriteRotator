import os
import cv2
import numpy as np

# Helper script to remove hidden RGB channels from images behind the alpha channel.
# Feel free to use this in your pipeline to ensure images are clean before processing.

class HiddenRGBRemover:
    def __init__(self):
        pass

    def remove_hidden_rgb(self, image):
        if image is None:
            print("Error: Could not load image")
            return

        # Ensure the image has an alpha channel
        alter = image.copy()
        if image.shape[2] == 4:
            # Set all pixels with alpha 0 to black (BGR)
            mask = image[:, :, 3] == 0
            image[mask] = 0  # Set BGR to black and keep alpha
        different = np.sum(image != alter)
        if different > 0:
            # print("No hidden RGB channels found, image is already clean.")
            return image, different
        return image, None

if __name__ == "__main__":
    # Run this as a script to process all images in a directory (in place or to a new directory)
    input_dir = os.path.join(os.path.dirname(__file__), "../image_bank", "variants_2")  # Directory of input images
    output_dir = os.path.join(os.path.dirname(__file__), "../image_bank", "variants_2")  # same if in place

    remover = HiddenRGBRemover()
    attempted, processed = 0, 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            processed_image, changed = remover.remove_hidden_rgb(image)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, processed_image)
            attempted += 1
            if changed is not None:
                processed += 1
                print(f"Processed image saved as: {output_path}. {changed} pixels changed. Attempted: {attempted}, Processed: {processed}")
            else:
                print(f"No changes made to {filename}, image is already clean. Attempted: {attempted}, Processed: {processed}")