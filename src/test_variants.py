import os
import csv
import cv2
import numpy as np
import random

from dataset_generation.labelling_app import Transform, LabellingApp

input_dir = os.path.join(os.path.dirname(__file__), f"../image_bank/{input('Enter the variant folder (e.g. variants): ')}")
labeller = LabellingApp()
# Test first: create a variant and make sure the new transform is correct
with open(input("Enter the path to the labelled CSV file: "), 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader, None)  # Skip header row
    for test_row in reader:  # Get the first row for testing
        test_row = next(reader)  # Get the first row for testing
        test_filename = test_row[0]
        print(f"test filename: {test_filename}")
        test_transform = Transform(flip=bool(int(test_row[1])), rot=float(test_row[2]), scale=float(test_row[3]))
        print(f"test transform: {test_transform}")
        test_image_path = os.path.join(input_dir, test_filename)

        if not os.path.exists(test_image_path):
            print("skipping this image")
            continue

        variant_image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)
        corrected_image = labeller.apply_transform(variant_image, test_transform)
        print(f"{corrected_image.shape=}, {test_transform=}")

        cv2.imshow("Variant Image", corrected_image)
        cv2.waitKey(0)
