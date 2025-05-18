from diffusers import Swin2SRPipeline
import torch
from PIL import Image

# Load sample image
image = Image.open("low_res_input.jpg").convert("RGB")

# Load the pretrained pipeline
pipeline = Swin2SRPipeline.from_pretrained(
    "caidas/swin2sr-lightweight-x2-64",
    torch_dtype=torch.float32
)

# Send model to CPU or GPU
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Upscale the image
with torch.no_grad():
    output = pipeline(image=image)["output"]

# Save the output
output.save("upscaled_output.jpg")
print("Upscaled image saved!")
