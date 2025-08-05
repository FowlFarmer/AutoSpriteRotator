# This script takes all your examples and creates variants of them by flipping, rotating, scaling and color adjusting them.
import os
import csv
import cv2
import numpy as np
import random

from labelling_app import Transform, LabellingApp



class VariantCreator:
    def __init__(self):
        self.labeller = LabellingApp()

    def create_variants(self, labelled_csv, input_dir="../image_bank/stripped/", output_dir="../image_bank/variants", new_csv_path="../datasets/variants_labels.csv"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(os.path.dirname(__file__), new_csv_path)):
            with open(os.path.join(os.path.dirname(__file__), new_csv_path), 'w', newline='') as new_csvfile:
                writer = csv.writer(new_csvfile)
                writer.writerow(['filename', 'flip', 'rotation_radians', 'scale'])

        with open(labelled_csv, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)  # Skip header row
            for row in reader:
                filename = row[0]
                transform_state = Transform(flip=bool(int(row[1])), rot=float(row[2]), scale=float(row[3]))

                # Load the original image
                image_path = os.path.join(os.path.dirname(input_dir), filename)
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    print(f"Error: Could not load image {image_path}")
                    continue

                # Create variants
                for i in range(20):  # Create 20 variants
                    variant_filename = filename.replace('stripped_', '').replace('.png', f'_v{i}.png')
                    if os.path.exists(os.path.join(output_dir, variant_filename)):
                        print(f"Variant {variant_filename} already exists, skipping.")
                        continue
                    variant_image, variant_transform = self.randomize_transform_and_color(image, transform_state)
                    cv2.imwrite(os.path.join(output_dir, variant_filename), variant_image)
                    
                    # Save the new labels
                    with open(os.path.join(os.path.dirname(__file__), new_csv_path), 'a', newline='') as new_csvfile:
                        writer = csv.writer(new_csvfile)
                        writer.writerow([variant_filename, int(variant_transform.flip), variant_transform.rot, variant_transform.scale])
                    print(f"Created variant {variant_filename} with transform {variant_transform}")

    def randomize_transform_and_color(self, image, transform_state):
        """Randomize an added transform, them apply this over the existing labelled transform_state,
            returning the adjusted transform_state and the image after adding the ADDED transform.
        
        """
        randomized_image = image.copy()
        new_flip = transform_state.flip
        new_rot = transform_state.rot
        new_scale = transform_state.scale

        # Randomly flip
        added_flip = False
        added_rot = 0.0
        added_scale = 1.0

        added_flip = random.randint(0, 1) == 1
        new_flip = new_flip ^ added_flip  # XOR to combine flips

        angle = random.uniform(-180, 180)  # degrees
        added_rot = np.radians(angle)
        # Keep rotation within -pi to pi
        new_rot -= added_rot # if I variate the rotation, I must subtract the added rotation (opposite)
        if new_rot > np.pi:
            new_rot -= 2 * np.pi
            print(f"new_rot > pi, adjusting: {new_rot}")
        elif new_rot < -np.pi:
            new_rot += 2 * np.pi
            print(f"new_rot < -pi, adjusting: {new_rot}")

        # Randomly scale
        added_scale = np.random.uniform(0.7, 1)
        new_scale /= added_scale # same thing, i need to negatively compensate

        randomized_image = self.labeller.apply_transform_reverse_order(randomized_image, Transform(flip=added_flip, rot=added_rot, scale=added_scale))

        # Randomly adjust color
        # don't affect alpha channel
        adjust_b = random.uniform(0.6, 1.5)
        adjust_g = random.uniform(0.6, 1.5)
        adjust_r = random.uniform(0.6, 1.5)

        # Apply adjustments with proper clipping
        randomized_image[:, :, 0] = np.clip(randomized_image[:, :, 0] * adjust_b, 0, 255).astype(randomized_image.dtype)
        randomized_image[:, :, 1] = np.clip(randomized_image[:, :, 1] * adjust_g, 0, 255).astype(randomized_image.dtype)
        randomized_image[:, :, 2] = np.clip(randomized_image[:, :, 2] * adjust_r, 0, 255).astype(randomized_image.dtype)
        print(f"variating flip: {added_flip}, rotation: {added_rot}, scale: {added_scale} color: B{adjust_b} G{adjust_g} R{adjust_r}")
        return randomized_image, Transform(flip=new_flip, rot=new_rot, scale=new_scale)

if __name__ == "__main__":
    labelled_csv = os.path.join(os.path.dirname(__file__), "../datasets/labels.csv")
    input_dir = os.path.join(os.path.dirname(__file__), "../image_bank/stripped/")
    output_dir = os.path.join(os.path.dirname(__file__), "../image_bank/variants/")
    new_csv_path = "../datasets/variants_labels.csv"

    creator = VariantCreator()

    if input("Test variant creation? (y/n): ").strip().lower() == 'y':
        # Test first: create a variant and make sure the new transform is correct
        with open(labelled_csv, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)  # Skip header row
            test_row = next(reader)  # Get the first row for testing
            test_filename = test_row[0]
            print(f"test filename: {test_filename}")
            test_transform = Transform(flip=bool(int(test_row[1])), rot=float(test_row[2]), scale=float(test_row[3]))
            print(f"test transform: {test_transform}")
            test_image_path = os.path.join(input_dir, test_filename)

            variant_image, variant_transform = creator.randomize_transform_and_color(cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED), test_transform)
            # Then apply the new transform and see if it gets us to the labelled correct rotation
            corrected_image = creator.labeller.apply_transform(variant_image, variant_transform)
            print(f"{corrected_image.shape=}, {variant_transform=}")
            cv2.imwrite(os.path.join(os.path.dirname(__file__), "../image_bank/test", f"corrected.png"), corrected_image)
            cv2.imwrite(os.path.join(os.path.dirname(__file__), "../image_bank/test", f"variant.png"), variant_image)

            original_image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)
            
            nonvariant = creator.labeller.apply_transform(original_image, test_transform)
            cv2.imwrite(os.path.join(os.path.dirname(__file__), "../image_bank/test", f"nonvariant.png"), nonvariant)
            cv2.imshow("Variant Image", corrected_image)
    else:
        creator.create_variants(labelled_csv, input_dir, output_dir, new_csv_path)