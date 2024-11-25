import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import (
    DDIMScheduler,
    DDIMInverseScheduler,
    StableDiffusionPix2PixZeroPipeline,
)
from proxy_config import set_proxy
from PIL import Image
import os

# 设置代理
# set_proxy()
device = "cuda:0"

# 加载BLIP模型和处理器
captioner_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(captioner_id)
model = BlipForConditionalGeneration.from_pretrained(
    captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
)

# 加载Stable Diffusion模型
sd_model_ckpt = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    sd_model_ckpt,
    caption_generator=model,
    caption_processor=processor,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipeline = pipeline.to(device)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)


# 定义图像处理函数
def process_image(image_path):
    raw_image = Image.open(image_path).resize((512, 512))
    caption = pipeline.generate_caption(raw_image)
    print(f"Caption for {image_path}: {caption}")

    generator = torch.manual_seed(0)
    inv_latents = pipeline.invert(caption, image=raw_image, generator=generator).latents

    source_prompts = caption
    target_prompts = "a photo of"

    source_embeds = pipeline.get_embeds(source_prompts, batch_size=2)
    target_embeds = pipeline.get_embeds(target_prompts, batch_size=2)

    edited_image = pipeline(
        caption,
        source_embeds=source_embeds,
        target_embeds=target_embeds,
        num_inference_steps=50,
        cross_attention_guidance_amount=0.15,
        generator=generator,
        latents=inv_latents,
        negative_prompt=caption,
    ).images[0]

    return edited_image


# 定义循环处理文件夹中所有图像的函数
def process_images_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            edited_image = process_image(image_path)
            output_path = os.path.join(output_folder, f"edited_{filename}")
            edited_image.save(output_path, format="png")
            print(f"Saved edited image to {output_path}")


def process_images_in_mfolder(fold_mapping):
    for input_folder, output_folder in fold_mapping.items():
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(input_folder, filename)
                edited_image = process_image(image_path)
                output_path = os.path.join(output_folder, f"edited_{filename}")
                edited_image.save(output_path, format="png")
                print(f"Saved edited image to {output_path}\n")


# 处理文件夹中的所有图像
# input_folder = "/lab/kirito/data/CNNspot_test/train/progan/airplane/0_real"
# output_folder = "/lab/kirito/data/CNNspot_test/train/progan/airplane/1_noisy"
fold_mapping = {
    "/lab/kirito/data/CNNspot_test/train/progan/bicycle/0_real": "/lab/kirito/data/CNNspot_test/train/progan/bicycle/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/bird/0_real": "/lab/kirito/data/CNNspot_test/train/progan/bird/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/boat/0_real": "/lab/kirito/data/CNNspot_test/train/progan/boat/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/bottle/0_real": "/lab/kirito/data/CNNspot_test/train/progan/bottle/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/bus/0_real": "/lab/kirito/data/CNNspot_test/train/progan/bus/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/car/0_real": "/lab/kirito/data/CNNspot_test/train/progan/car/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/cat/0_real": "/lab/kirito/data/CNNspot_test/train/progan/cat/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/chair/0_real": "/lab/kirito/data/CNNspot_test/train/progan/chair/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/cow/0_real": "/lab/kirito/data/CNNspot_test/train/progan/cow/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/diningtable/0_real": "/lab/kirito/data/CNNspot_test/train/progan/diningtable/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/dog/0_real": "/lab/kirito/data/CNNspot_test/train/progan/dog/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/horse/0_real": "/lab/kirito/data/CNNspot_test/train/progan/horse/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/motorbike/0_real": "/lab/kirito/data/CNNspot_test/train/progan/motorbike/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/person/0_real": "/lab/kirito/data/CNNspot_test/train/progan/person/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/pottedplant/0_real": "/lab/kirito/data/CNNspot_test/train/progan/pottedplant/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/sheep/0_real": "/lab/kirito/data/CNNspot_test/train/progan/sheep/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/sofa/0_real": "/lab/kirito/data/CNNspot_test/train/progan/sofa/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/train/0_real": "/lab/kirito/data/CNNspot_test/train/progan/train/1_noisy",
    "/lab/kirito/data/CNNspot_test/train/progan/tvmonitor/0_real": "/lab/kirito/data/CNNspot_test/train/progan/tvmonitor/1_noisy",
}
# process_images_in_folder(input_folder, output_folder)
process_images_in_mfolder(fold_mapping)
