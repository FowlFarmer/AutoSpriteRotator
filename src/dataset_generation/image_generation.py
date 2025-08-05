from diffusers import StableDiffusionXLPipeline
import cv2
import numpy as np
import torch
import time
import os

class ImageGenerator:
    def __init__(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe.to("cuda")
        self.neg_prompt = "ugly, blurry, poor quality, complex background, multiple objects, concept art, two objects, three objects, person, human" # Negative prompt here

    def generate_image(self, prompt):
        image = self.pipe(prompt=prompt, negative_prompt=self.neg_prompt).images[0]
        print("PIL type:", type(image))  # Should be PIL.Image.Image
        image = image.convert("RGB")          # 🟢 Ensure RGB mode
        image_np = np.array(image)
        output = cv2.cvtColor(src=image_np, code=cv2.COLOR_RGB2BGR)
        return output

    def test(self):
        print("CUDA available:", torch.cuda.is_available())
        print(np.__version__)
        # to cuda
        while True:
            prompt = input("Enter your prompt: ")
            output = self.generate_image(prompt)
            # try:
            #     cv2.imshow("Generated Image", output)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            # except Exception as e:
            #     print("Error displaying image:", e)

            # Generate dynamic filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            directory = input("Enter the directory to save the image: ")
            output_name = os.path.join(directory, timestamp + ".png")
            txt_name = os.path.join(directory, timestamp + ".txt")

            # Save image
            cv2.imwrite(output_name, output)
            print(f"Image saved as: {output_name}")
            
            # Save prompt to text file
            # with open(txt_name, 'w') as f:
            #     f.write(f"Prompt: {prompt}\n")
            #     f.write(f"Negative Prompt: {neg_prompt}\n")
            #     f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"Prompt saved as: {txt_name}")



if __name__ == "__main__":
    generator = ImageGenerator()
    generator.test()
