from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import torch
from proxy_config import set_proxy

# 设置代理
set_proxy()


def simple_label(label):
    label_map = {
        "airplane": "plane",
        "motorcycle": "bike",
        "fire hydrant": "hydrant",
        "hot dog": "hotdog",
    }
    return label_map.get(label, label.split()[-1])

def predict_image_from_url(url):
    # Load image from URL
    image = Image.open(url)

    # Initialize Sreekanth's processor and model
    processor = AutoImageProcessor.from_pretrained(
        "Sreekanth3096/vit-coco-image-classification"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "Sreekanth3096/vit-coco-image-classification"
    )

    # Preprocess image and make predictions
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Get predicted class label
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    
    simple_class = simple_label(predicted_class)

    return predicted_class


# Example usage
if __name__ == "__main__":
    url = "/lab/kirito/data/CNNspot_test/test/cyclegan/apple/1_fake/n07749192_721_fake.png"
    predicted_class = predict_image_from_url(url)
    print(f"Predicted class: {predicted_class}")
