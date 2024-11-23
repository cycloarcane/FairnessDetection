import math
import os
import torch.nn as nn
import logging
import torch.nn.functional as F
import torch
import numpy as np
from sklearn import metrics
from metrics.base_metrics_class import calculate_metrics_for_train
from loss import LOSSFUNC
from detectors import DETECTOR
from clip import clip
# from training.utils.utils import compute_accuracy

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logger = logging.getLogger(__name__)

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

'''CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768
}'''


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    model = clip.build_model(state_dict or model.state_dict())

    return model


class LinearClassifier(torch.nn.Module):
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        # torch.set_default_dtype(torch.float16)
        self.num_labels = num_labels
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)

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
        temp = CUSTOM_TEMPLATES['biggan']
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model ):
        super().__init__()
        self.image_encoder = clip_model.visual
        # self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.vision_clip = clip_model.visual
        # self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter1 = Adapter(1024, 4).to(clip_model.dtype)
        self.adapter2 = Adapter(1024, 4).to(clip_model.dtype)
        self.channel = FeatureFusionModule()
        self.fc = LinearClassifier(768, 2)
        

    def forward(self, image, phase):
        if phase == 'train':
            image_features1 = self.image_encoder(image['fake_images'].type(self.dtype)) 
            x = self.adapter1(image_features1)
            ratio = 0.4

            image_features1 = ratio * x + (1 - ratio) * image_features1
            image_features2 = self.vision_clip(image['real_images'].type(self.dtype))

            image_features = torch.cat((image_features1, image_features2), 1).unsqueeze(-1).unsqueeze(-1)

            final_features = self.channel(image_features)
            
            final_features_adapter = self.adapter2(image_features1.detach())
            final_features_adapter = ratio * final_features_adapter + (1 - ratio) * image_features1
            
            logits1 = self.fc(final_features)
            logits2 = self.fc(final_features_adapter)

            # text_features = self.text_encoder()

            # image_features1 = final_features / final_features.norm(dim=-1, keepdim=True)
            # image_features2 = final_features_adapter / final_features_adapter.norm(dim=-1, keepdim=True)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # logit_scale = self.logit_scale.exp()
            # logits1 = logit_scale * image_features1 @ text_features.t()
            # logits2 = logit_scale * image_features2 @ text_features.t()

            return logits1,logits2
        
        elif phase == 'val':
            image_features = self.image_encoder(image['image'].type(self.dtype))
            temp1 = image_features
            x = self.adapter1(image_features)
            ratio = 0.4
            image_features = ratio * x + (1 - ratio) * image_features

            image_features_fin = self.adapter2(image_features)
            image_features_fin = ratio * image_features_fin + (1 - ratio) * image_features
            temp2 = image_features_fin
            
            logits = self.fc(image_features_fin)

            # text_features = self.text_encoder()

            # image_features = image_features_fin / image_features_fin.norm(dim=-1, keepdim=True)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # logit_scale = self.logit_scale.exp()
            # logits = logit_scale * image_features_fin @ text_features.t()
            prob_tmp = torch.softmax(logits, dim=1)[:, 1]

            pred_dict = {'cls': logits, 'prob': prob_tmp, 'clip_f': temp1, 'adapter_f': temp2}
            
            return pred_dict


@DETECTOR.register_module(module_name='CLIP_Adapter_rec')
class CLIP_Adapter_rec(nn.Module):
    def __init__(self, cfg):
        super(CLIP_Adapter_rec, self).__init__()
        self.cfg = cfg
        self.loss_func = self.build_loss()
        classnames = ['fake','real']
        print(f'Loading CLIP (backbone: {self.cfg.backbone})')
        clip_model = load_clip_to_cpu(self.cfg)
        clip_model.float()

        print('Building custom CLIP')

        self.model = CustomCLIP(self.cfg, classnames, clip_model)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def get_test_metrics(self):
        '''self.prob = np.expand_dims(self.prob,axis=0)
        self.label = np.expand_dims(self.label,axis=0)'''
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        # eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true,y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {'acc':acc, 'auc':auc, 'ap':ap, 'pred':y_pred, 'label':y_true}  

    def forward(self, x, phase):
        if phase == 'train':
            predict1, predict2 = self.model(x, phase)
        
            return predict1,predict2
        
        elif phase == 'val':
            predict = self.model(x, phase)
            self.prob.append(
                predict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                x['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )

            # deal with acc
            _, prediction_class = torch.max(predict['cls'], 1)
            correct = (prediction_class == x['label']).sum().item()
            self.correct += correct
            self.total += x['label'].size(0)
            return predict
    

    def build_loss(self):
        # prepare the loss function
        loss_class = LOSSFUNC['cross_entropy']
        loss_func = loss_class()  # use am-softmax for srm, params are specified by the author
        return loss_func
    
    def get_losses(self, data_dict, pred1, pred2):
        label = data_dict['fake_labels']
        loss1 = self.loss_func(pred1, label)
        loss2 = self.loss_func(pred2, label)
        loss_dict = {'firstloss': loss1,
                     'finloss':loss2} 
        return loss_dict
    
    def get_train_metrics(self, data_dict, pred):
        label = data_dict['fake_labels']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    


class Channel_Attention_Module_Conv(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(Channel_Attention_Module_Conv, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x).view(x.size(0), x.size(1), -1)  # Shape: [N, C, 1]
        max_x = self.max_pooling(x).view(x.size(0), x.size(1), -1)  # Shape: [N, C, 1]
        avg_out = self.conv(avg_x).view(x.size(0), x.size(1), 1, 1)  # Shape: [N, C, 1, 1]
        max_out = self.conv(max_x).view(x.size(0), x.size(1), 1, 1)  # Shape: [N, C, 1, 1]
        v = self.sigmoid(avg_out + max_out)
        return v

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=1536, mid_chan=768, out_chan=768):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU()
        )
        self.sc = Channel_Attention_Module_Conv(in_chan)
        self.fc = nn.Linear(mid_chan, out_chan)  # 用于将特征恢复到所需的输出维度
        self.init_weight()

    def forward(self, fin_feature):
        Attsrm = self.sc(fin_feature)
        fuse = self.convblk(fin_feature + fin_feature * Attsrm)
        fuse = fuse.view(fuse.size(0), -1, fuse.size(2) * fuse.size(3)).mean(dim=-1)  # Flatten and average spatial dimensions
        fuse = self.fc(fuse)  # Apply fully connected layer to get desired output dimension
        return fuse  # Return the final output

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, nn.Linear):
                nn.init.xavier_normal_(ly.weight)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)




