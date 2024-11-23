import torch
import logging
from clip import clip
from PIL import Image
import torch.nn as nn
import numpy as np
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv
from sklearn import metrics
from metrics.base_metrics_class import calculate_metrics_for_train
from loss import LOSSFUNC
from detectors import DETECTOR
from clip import clip
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

CHANNELS = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/16": 768 + 512}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class CLIPModel(nn.Module):
    def __init__(self, name="ViT-L/14", num_classes=1):
        super(CLIPModel, self).__init__()
        # self.trainer = trainer
        self.model, self.preprocess = clip.load(
            name, device="cpu"
        )  # self.preprecess will not be used during training, which is handled in Dataset class
        self.gcn_model = GCNModel(CHANNELS[name], 512, 50)
        self.fc = nn.Linear(CHANNELS[name] + 50, num_classes)

    def forward(self, x, label, return_feature=False):
        # print(x.shape)
        self.f_g1 = self.model.encode_image(x)
        # self.f_g1 = torch.cat([f, f_proj], dim=-1)
        if return_feature:
            return self.f_g1
        self.label = label
        self.edge_index, self.loss_index = self.generate_edge_index(
            self.f_g1, self.label
        )
        self.gcn_output = self.gcn_model(self.f_g1, self.edge_index)
        self.f_g = torch.cat([self.f_g1, self.gcn_output], dim=-1)
        self.output = self.fc(self.f_g)
        return self.output, self.f_g, self.loss_index, self.f_g1

    def generate_edge_index(self, features, labels):
        num_nodes = len(labels)
        edge_index = []
        # 计算节点之间的相似度
        loss_index = []
        similarity_matrix = torch.mm(features, features.transpose(0, 1))
        # 使用广播机制创建一个 4x4 的矩阵，每行都是 labels_tensor
        matrix = labels.unsqueeze(0).expand(num_nodes, -1)

        transposed_matrix = torch.transpose(matrix, 0, 1)

        boolean_matrix = matrix != transposed_matrix

        sum = torch.sum(similarity_matrix[boolean_matrix])
        count_true = torch.sum(boolean_matrix).item()

        if count_true > 0:
            threshold = sum.item() / count_true
        else:
            threshold = 0.0

        # 基于某个阈值构建边索引
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if labels[i] == labels[j]:
                    similarity = similarity_matrix[i, j].item()
                    if similarity <= threshold:
                        edge_index.append([i, j])
                        edge_index.append([j, i])
                        loss_index.append([i, j])

        # 转换成PyTorch Geometric所需的格式，并确保是无向图
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_index = edge_index.cuda()

        return edge_index, loss_index


@DETECTOR.register_module(module_name="CLIP_GCN_test")
class CLIP_GCN_test(nn.Module):
    def __init__(self):
        super(CLIP_GCN_test, self).__init__()
        self.loss_func = self.build_loss()
        self.model = CLIPModel()

        print("Building custom CLIP")

        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def get_test_metrics(self):
        """self.prob = np.expand_dims(self.prob,axis=0)
        self.label = np.expand_dims(self.label,axis=0)"""
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
        acc = metrics.accuracy_score(y_true, y_pred >= 0.5)
        # acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {"acc": acc, "auc": auc, "ap": ap, "pred": y_pred, "label": y_true}

    def forward(self, x, inference=False):
        # get the prediction by classifier
        pred, f_g, loss_index, f_g1 = self.model(x["image"], x["label"])
        # get the probability of the pred
        # prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {
            "cls": pred,
            "f_g": f_g,
            "loss_index": loss_index,
            "f_g1": f_g1,
        }
        if inference:
            self.prob.append(pred_dict["cls"].detach().squeeze().cpu().numpy())
            self.label.append(x["label"].detach().squeeze().cpu().numpy())
            # deal with acc
            _, prediction_class = torch.max(pred, 1)
            correct = (prediction_class == x["label"]).sum().item()
            self.correct += correct
            self.total += x["label"].size(0)
        return pred_dict

    def build_loss(self):
        # prepare the loss function
        loss_class = LOSSFUNC["gcn_loss"]
        loss_func = (
            loss_class()
        )  # use am-softmax for srm, params are specified by the author
        return loss_func

    def get_losses(self, data_dict, pred_dict):
        label = data_dict["label"]
        pred = pred_dict["cls"]
        f_g = pred_dict["f_g"]
        loss_index = pred_dict["loss_index"]
        loss = self.loss_func(pred, f_g, loss_index, label.float())
        loss_dict = {"overall": loss}
        return loss_dict

    def get_train_metrics(self, data_dict, pred):
        label = data_dict["label"]
        pred = pred["cls"]
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        # acc = accuracy_score(label.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy() >= 0.5)
        metric_batch_dict = {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
        return metric_batch_dict
