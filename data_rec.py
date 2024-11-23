from diffusers import AutoPipelineForImage2Image,AutoPipelineForText2Image
import torch
import csv
from proxy_config import set_proxy
set_proxy()
import sys
import os
import PIL.Image as Image
sys.path.append("/lab/kirito/data/sd-inpainting")
# from PromptBook.get_promptItemFromPromptBook import get_random_prompt

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline = AutoPipelineForImage2Image.from_pipe(
    pipeline_text2image
).to("cuda")

# save_folder = "/home/ubuntu/anahera/VQA-Deepfake-Dataset-XinanHe/DF_VQA_Dataset/Not_exist/SDXL/train/T2I"
# dir_csv = "/home/ubuntu/anahera/VQA-Deepfake-Dataset-XinanHe/DF_VQA_Dataset/Not_exist/SDXL/train.csv"

# with open(dir_csv, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(['image','prompt','is_fake','forgery_type','pattern'])
save_folder = "/lab/kirito/data/CNNspot_test/train/progan/person/tmp"

# prompt = get_random_prompt()
image = Image.open(f'/lab/kirito/data/CNNspot_test/train/progan/person/0_real/00007.png').convert('RGB').resize((768, 768))
prompt = "a photo of person,like a original image"
image = pipeline(prompt = prompt,image = image, strength = 0.8, guidance_scale = 1.5, image_guidance_scale = 10.5).images[0]

save_path = os.path.join(save_folder, f'00007_p.png')

image.save(save_path)