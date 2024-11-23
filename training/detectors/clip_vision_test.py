import os
import torch.nn as nn
import logging
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


class clipmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor, self.preprocess = clip.load("ViT-L/14", device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class
        # self.fc = nn.Linear(768, 2)
        # self.adapter = Adapter(1024, 4).to(self.feature_extractor.dtype)
        self.fc = LinearClassifier(768, 2)

    def forward(self, x):
        # with torch.no_grad():
        intermediate_output = self.feature_extractor.encode_image(x)
        # intermediate_output = self.adapter(intermediate_output)
        output = self.fc(intermediate_output)
        return output


@DETECTOR.register_module(module_name='CLIP_Vision_test')
class CLIP_Vision_test(nn.Module):
    def __init__(self):
        super(CLIP_Vision_test, self).__init__()
        self.loss_func = self.build_loss()
        self.model = clipmodel()

        print('Building custom CLIP')


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
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {'acc': acc, 'auc': auc, 'ap': ap, 'pred': y_pred, 'label': y_true}

    def forward(self, x, inference=False):
        # get the prediction by classifier
        pred = self.model(x['image'])
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob}
        if inference:
            self.prob.append(
                pred_dict['prob']
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
            _, prediction_class = torch.max(pred, 1)
            correct = (prediction_class == x['label']).sum().item()
            self.correct += correct
            self.total += x['label'].size(0)
        return pred_dict

    def build_loss(self):
        # prepare the loss function
        loss_class = LOSSFUNC['cross_entropy']
        loss_func = loss_class()  # use am-softmax for srm, params are specified by the author
        return loss_func

    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict, pred):
        label = data_dict['label']
        pred = pred['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
