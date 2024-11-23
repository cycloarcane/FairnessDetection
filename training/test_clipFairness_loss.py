import os
import argparse
from detectors import DETECTOR
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import datetime
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision
from dataset.mydataset_test import testDataset
import csv
import time
from sklearn.metrics import log_loss, roc_auc_score
import multiprocessing
import pickle
import numpy as np
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, average_precision_score

torch.cuda.set_device(1)
print()

clip_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}

def eff_pretrained_model(numClasses):
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = nn.Linear(1536, numClasses)
    return model

def classification_metrics(label, prediction):
    auc = roc_auc_score(label, prediction)
    CM = confusion_matrix(label, prediction >= 0.5)
    acc = accuracy_score(label, prediction >= 0.5)
    ap = average_precision_score(label, prediction)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    return auc, ap, TPR, FPR, acc

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_paths", type=str, nargs='+',
                        default=["/lab/kirito/data/CNNspot_test/test/progan/airplane_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/bicycle_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/bird_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/boat_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/bottle_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/bus_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/car_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/cat_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/chair_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/cow_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/diningtable_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/dog_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/horse_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/motorbike_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/person_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/pottedplant_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/sheep_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/sofa_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/train_test.csv",
                                 "/lab/kirito/data/CNNspot_test/test/progan/tvmonitor_test.csv"])
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int,
                        default=64, help="size of the batches")
    parser.add_argument("--num_out_classes", type=int, default=1)
    parser.add_argument("--checkpoints", type=str,
                        default="/lab/kirito/clip-fairness/checkpoints/clip-adapter_small_loss/progan_lr0.001/clip-adapter_small_loss19.pth")
    parser.add_argument("--model_structure", type=str, default='clip-adapter_small_loss',
                        help="efficient,ucf_daw")
    parser.add_argument("--backbone", type=str, default="ViT-L/14", help="name of CNN backbone")

    opt = parser.parse_args()
    print(opt, '!!!!!!!!!!!')

    cuda = True if torch.cuda.is_available() else False

    # prepare the model (detector)
    if opt.model_structure == 'clip-adapter_small_loss':
        model_class = DETECTOR['CLIP_Fairness_loss']
        # print(model_class)

    if opt.model_structure == 'clip-adapter_small_loss':
        model = model_class(opt)
    if cuda:
        # model.cuda()
        model.to(device)

    ckpt = torch.load(opt.checkpoints, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    print('loading from: ', opt.checkpoints)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    for test_path in opt.test_paths:
        test_dataset = testDataset(test_path, clip_data_transforms['test'])
        test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

        print("%s" % test_path)
        print('-' * 10)
        print('%d batches in total' % len(test_dataloader))

        corrects = 0.0

        pred_label_list = []
        pred_probs_list = []
        label_list = []
        total = 0
        running_corrects = 0

        for i, data_dict in enumerate(test_dataloader):
            bSTime = time.time()
            phase = 'val'
            model.eval()
            data, label = data_dict['image'], data_dict["label"]
            data_dict['image'], data_dict["label"] = data.to(device), label.to(device)
            data = data.to(device)
            labels = torch.where(data_dict['label'] != 0, 1, 0)
            with torch.no_grad():
                preds = model(data_dict, phase)
                _, preds_label = torch.max(preds['cls'], 1)

                pred_probs = torch.softmax(preds['cls'], dim=1)[:, 1]
                total += data_dict['label'].size(0)
                running_corrects += (preds_label == data_dict['label']).sum().item()
                p = (preds_label == data_dict['label']).sum().item()
                print(p)
                preds_label = preds_label.cpu().data.numpy().tolist()
                pred_probs = pred_probs.cpu().data.numpy().tolist()

            pred_label_list += preds_label
            pred_probs_list += pred_probs
            label_list += label.cpu().data.numpy().tolist()
            bETime = time.time()
            print('#{} batch finished, elapsed time: {}'.format(i, bETime - bSTime))

        pred_label_list = np.array(pred_label_list)
        pred_probs_list = np.array(pred_probs_list)
        label_list = np.array(label_list)

        epoch_acc = running_corrects / total

        auc, ap, TPR, FPR, acc = classification_metrics(label_list, pred_probs_list)

        print('Results for {}:'.format(test_path))
        print('Acc: {:.4f},  AP: {:.4f},  AUC: {:.4f},  TPR: {:.4f},  FPR: {:.4f}'.format(
            epoch_acc, ap, auc, TPR, FPR))
        print()
        print('-' * 10)
