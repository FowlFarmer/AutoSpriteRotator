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
        """Apply the current transform to the image on a padded canvas, then crop/resize back."""
        original_h, original_w = img.shape[:2]

        # Create a double-size canvas (with transparency if needed)
        canvas_h = original_h * 3
        canvas_w = original_w * 3
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        # Compute top-left coordinates to center the image in the canvas
        offset_y = (canvas_h - original_h) // 2
        offset_x = (canvas_w - original_w) // 2

        # Place original image at center
        canvas[offset_y:offset_y+original_h, offset_x:offset_x+original_w] = img
        result = canvas

        center = (canvas_w // 2, canvas_h // 2)

        # Apply flip
        if transform_state.flip:
            result = cv2.flip(result, 1)  # horizontal flip

        # Apply scaling
        if transform_state.scale != 1.0:
            scaled_w = int(canvas_w * transform_state.scale)
            scaled_h = int(canvas_h * transform_state.scale)
            result = cv2.resize(result, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            # Crop or pad to canvas size again
            if transform_state.scale > 1.0:
                crop_x = (scaled_w - canvas_w) // 2
                crop_y = (scaled_h - canvas_h) // 2
                result = result[crop_y:crop_y+canvas_h, crop_x:crop_x+canvas_w]
            else:
                padded = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
                pad_x = (canvas_w - scaled_w) // 2
                pad_y = (canvas_h - scaled_h) // 2
                padded[pad_y:pad_y+scaled_h, pad_x:pad_x+scaled_w] = result
                result = padded

        # Apply rotation
        if transform_state.rot != 0:
            rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(transform_state.rot), 1.0)
            result = cv2.warpAffine(result, rotation_matrix, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        # Final crop back to original size
        final_crop = result[offset_y:offset_y+original_h, offset_x:offset_x+original_w]

        return final_crop

    def apply_transform_reverse_order(self, img, transform_state):
        """Apply the current transform to the image on a padded canvas, then crop/resize back."""
        original_h, original_w = img.shape[:2]

        # Create a double-size canvas (with transparency if needed)
        canvas_h = original_h * 3
        canvas_w = original_w * 3
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        # Compute top-left coordinates to center the image in the canvas
        offset_y = (canvas_h - original_h) // 2
        offset_x = (canvas_w - original_w) // 2

        # Place original image at center
        canvas[offset_y:offset_y+original_h, offset_x:offset_x+original_w] = img
        result = canvas

        center = (canvas_w // 2, canvas_h // 2)

        # Apply rotation
        if transform_state.rot != 0:
            rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(transform_state.rot), 1.0)
            result = cv2.warpAffine(result, rotation_matrix, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        # Apply scaling
        if transform_state.scale != 1.0:
            scaled_w = int(canvas_w * transform_state.scale)
            scaled_h = int(canvas_h * transform_state.scale)
            result = cv2.resize(result, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            # Crop or pad to canvas size again
            if transform_state.scale > 1.0:
                crop_x = (scaled_w - canvas_w) // 2
                crop_y = (scaled_h - canvas_h) // 2
                result = result[crop_y:crop_y+canvas_h, crop_x:crop_x+canvas_w]
            else:
                padded = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
                pad_x = (canvas_w - scaled_w) // 2
                pad_y = (canvas_h - scaled_h) // 2
                padded[pad_y:pad_y+scaled_h, pad_x:pad_x+scaled_w] = result
                result = padded

        # Apply flip
        if transform_state.flip:
            result = cv2.flip(result, 1)  # horizontal flip

        # Final crop back to original size
        final_crop = result[offset_y:offset_y+original_h, offset_x:offset_x+original_w]

        return final_crop


# Example usage
if __name__ == "__main__":
    label_output_path = os.path.join(os.path.dirname(__file__), "../../datasets", "labels.csv")
    images_dir = os.path.join(os.path.dirname(__file__), "../../image_bank", "stripped")  # Directory of input images
    transformed_dir = os.path.join(os.path.dirname(__file__), "../../image_bank", "transformed")  # Directory to save transformed images

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
            if transform.flip:
                flip = "1"
            else:
                flip = "0"
            writer.writerow([filename, flip, transform.rot, transform.scale])