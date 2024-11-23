import torch
import torch.nn as nn
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a {} image.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',

    'biggan': 'a {} photo.',
    'cyclegan': 'a {} photo.',
    'dalle2': 'a {} photo.',
    'deepfake': 'a {} photo.',
    'eg3d': 'a {} photo.',
    'gaugan': 'a {} photo.',
    'glide_50_27': 'a {} photo.',
    'glide_100_10': 'a {} photo.',
    'glide_100_27': 'a {} photo.',
    'guided': 'a {} photo.',
    'ldm_100': 'a {} photo.',
    'ldm_200': 'a {} photo.',
    'ldm_200_cfg': 'a {} photo.',
    'progan': 'a {} photo.',
    'sd_512x512': 'a {} photo.',
    'sdxl': 'a {} photo.',
    'stargan': 'a {} photo.',
    'stylegan': 'a {} photo.',
    'stylegan2': 'a {} photo.',
    'stylegan3': 'a {} photo.',
    'taming': 'a {} photo.',
    'firefly': 'a {} photo.',
    'midjourney_v5': 'a {} photo.',
    'dalle3': 'a {} photo.',
    'faceswap': 'a {} photo.',
    'progan_train': 'a {} photo.',
}


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 384, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(384, 768, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(1024, 4).to(clip_model.dtype)

    def forward(self, other_images, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)
        ratio = 0.4
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
