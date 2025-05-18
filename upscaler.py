# upscaler.py
import torch
from PIL import Image
from torchvision import transforms
from diffusers import Swin2SRPipeline
import os

# Load model only once (for performance)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "caidas/swin2SR-classical-sr-x4-64"
pipe = Swin2SRPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

def upscale_image(input_pil):
    input_pil = input_pil.convert("RGB")
    low_res = transforms.ToTensor()(input_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = pipe(image=low_res)["sample"]
    output_image = output.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((output_image * 255).astype("uint8"))
