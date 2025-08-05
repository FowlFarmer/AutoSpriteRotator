import cv2
import numpy as np
from typing import Tuple, NamedTuple
import csv
import os


class Transform(NamedTuple):
    flip: bool
    rot: float
    scale: float



class LabellingApp:
    def __init__(self):
        pass

    def process_key(self, key: int):
        match key:
            case 119:  # Up arrow
                return "w"
            case 97:  # Down arrow
                return "a"
            case 115:  # Left arrow
                return "s"
            case 100:  # Right arrow
                return "d"
            case 102:
                return "f"
            case 13:  # Enter key
                return "enter"
            case 27:  # ESC key
                return "esc"
            case 8:  # Backspace key
                return "backspace"
            case _:
                return None

    def interactive_image_labeller(self, image_path: str):
        """
        Opens an image and allows interactive adjustment of flip, rotation, and scale.
        
        Controls:
        - F: Toggle flip
        - Left/Right arrows: Rotate (hold for continuous, tap for 1 degree)
        - Up/Down arrows: Scale (hold for continuous, tap for small increment)
        - Enter: Save current transform and exit
        - ESC: Exit without saving
        """
        original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # Must be IMREAD_UNCHANGED to preserve alpha channel
        if original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
    
        # Initialize transform state
        transform = Transform(flip=False, rot=0.0, scale=1.0)

        while True:
            transformed_image = self.apply_transform(original_image, transform)
            cv2.imshow('Image Labeller', transformed_image)
            
            key = cv2.waitKey(15)
            if(not key or key == -1):  # No valid key pressed
                continue

            action = self.process_key(key)
            if action == "w":
                transform = transform._replace(scale=transform.scale + 0.05)
            elif action == "a":
                transform = transform._replace(rot=transform.rot + np.radians(4))
            elif action == "s":
                transform = transform._replace(scale=max(transform.scale - 0.05, 0.1))
            elif action == "d":
                transform = transform._replace(rot=transform.rot - np.radians(4))
            elif action == "f":
                transform = transform._replace(flip=not transform.flip)
            elif action == "enter":
                print(f"Transform latched: {transform}")
                return transform, transformed_image, True
            elif action == "esc":
                print("Exiting without saving transform.")
                return None, None, True # two nones for typing consistency
            elif action == "backspace":
                print("Backspace pressed, skipping.")
                return None, None, False  # Skip this image
            
    def apply_transform(self, img, transform_state):
        """Apply the current transform to the image"""
        result = img.copy()
        height = img.shape[0]
        width = img.shape[1]
        center = (width // 2, height // 2)

        # Apply flip
        if transform_state.flip:
            result = cv2.flip(result, 1)  # horizontal flip
        # Apply scale
        if transform_state.scale != 1.0:
            new_width = int(width * transform_state.scale)
            new_height = int(height * transform_state.scale)
            result = cv2.resize(result, (new_width, new_height))
            
            # Center the scaled image
            if transform_state.scale < 1.0:
                # If scaled down, pad with black
                pad_x = (width - new_width) // 2
                pad_y = (height - new_height) // 2
                padded = np.zeros((height, width, 4), dtype=np.uint8)
                padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = result
                result = padded
            elif transform_state.scale > 1.0:
                # If scaled up, crop from center
                crop_x = (new_width - width) // 2
                crop_y = (new_height - height) // 2
                result = result[crop_y:crop_y+height, crop_x:crop_x+width]
        
        # Apply rotation
        if transform_state.rot != 0:
            rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(transform_state.rot), 1.0)
            result = cv2.warpAffine(result, rotation_matrix, (width, height))
        
        
        return result

 
# Example usage
if __name__ == "__main__":
    label_output_path = os.path.join(os.path.dirname(__file__), "../datasets", "labels.csv")
    images_dir = os.path.join(os.path.dirname(__file__), "../image_bank", "stripped")  # Directory of input images
    transformed_dir = os.path.join(os.path.dirname(__file__), "../image_bank", "transformed")  # Directory to save transformed images

    labeller = LabellingApp()
    pregenerated_filenames = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    if not os.path.exists(label_output_path) or os.path.getsize(label_output_path) == 0:
        print(f"Label output file {label_output_path} does not exist or is empty, creating new file.")
        with open(label_output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'flip', 'rotation_radians', 'scale'])

    else: # Only check for pre-existing labels if the label file exists (duh)
        print("Checking for images in directory already labelled...")
        with open(label_output_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)  # Skip header row
            for row in reader:
                if row is None or len(row) < 1:
                    continue
                filename = row[0]
                if filename in pregenerated_filenames:
                    pregenerated_filenames.remove(filename)
                    print(f"Image {filename} already labelled, removing from pregenerated list.")

    if not os.path.exists(transformed_dir):
        print(f"Transformed images output directory {transformed_dir} does not exist, creating it.")
        os.makedirs(transformed_dir, exist_ok=True)

    print(f"Use WASD to rotate and scale images, F to flip, Enter to save, ESC to exit.")
    print(f"Beginning labelling. {len(pregenerated_filenames)} images to label. (Good luck! >w<)")
    for filename in pregenerated_filenames:
        image_path = os.path.join(images_dir, filename)
        transform, transformed_image, keep = labeller.interactive_image_labeller(image_path)
        if not keep:
            print(f"Skipping {filename}.")
            continue
        if transform is None or transformed_image is None:
            print(f"Quitting labelling for {filename}.")
            break

        output_name = filename.replace('stripped', 'transformed')
        output_path = os.path.join(transformed_dir, output_name)
        cv2.imwrite(output_path, transformed_image)
        print(f"Transform {transform} for {filename}")
        with open(label_output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, transform.flip, transform.rot, transform.scale])