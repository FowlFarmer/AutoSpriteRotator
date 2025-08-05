from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import time
import os

class BackgroundRemover:
    def __init__(self):
        # Load background removal model
        self.rmbg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self.rmbg_model.to('cuda')
        self.rmbg_model.eval()

        # Data settings for background removal
        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def remove_background(self, image):
        # Convert OpenCV image (BGR) back to PIL Image (RGB)
        pil_image = self.cv2_to_pil(image)

        # Prepare image for background removal model
        input_images = self.transform_image(pil_image).unsqueeze(0).to('cuda')

        # Prediction
        with torch.no_grad():
            preds = self.rmbg_model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(pil_image.size)
    
        # Apply alpha channel (transparency) using the mask
        pil_image.putalpha(mask)
        
        # Convert back to OpenCV format with transparency
        # Note: OpenCV doesn't handle RGBA well, so we'll return PIL image
        return pil_image
    
    def pil_to_cv2(self, pil_image):
        """Convert PIL Image with alpha channel to OpenCV format (BGRA) (THIS ONE TAKES ALPHA)"""
        image_rgba = np.array(pil_image.convert("RGBA"))
        return cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA)
    
    def cv2_to_pil(self, cv2_image):
        """Convert OpenCV image (BGR) to PIL Image (RGB) (THIS ONE DOESNT TAKE ALPHA)"""
        image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)

    def test(self):
        print("CUDA available:", torch.cuda.is_available())
        print(np.__version__)

        # Load an image
        image_path = input("(note: saves to this dir) Enter the path to the image: ")
        directory = os.path.dirname(image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return

        # Remove background
        pil_image = self.remove_background(image)
        
        # Convert back to OpenCV format
        output_image = self.pil_to_cv2(pil_image)

        # Display the result
        cv2.imshow("Background Removed Image", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_name = os.path.join(directory, timestamp + ".png")

        # Save image
        cv2.imwrite(output_name, output_image)
        print(f"Image saved as: {output_name}")


if __name__ == "__main__":
    print("Note: Using this model requires an access token from Hugging Face. Use the cli command `huggingface-cli login` to authenticate.")
    remover = BackgroundRemover()
    remover.test()

