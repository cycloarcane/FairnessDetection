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
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    average_precision_score,
)

torch.cuda.set_device(0)
print()

clip_data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    ),
}


def eff_pretrained_model(numClasses):
    model = EfficientNet.from_pretrained("efficientnet-b3")
    model._fc = nn.Linear(1536, numClasses)
    return model


def classification_metrics(label, prediction):
    auc = roc_auc_score(label, prediction)
    # auc = 0
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_paths",
        type=str,
        nargs="+",
        default=[
            "/lab/kirito/data/CNNspot_test/test/stylegan2/car_test.csv",
            "/lab/kirito/data/CNNspot_test/test/stylegan2/cat_test.csv",
            "/lab/kirito/data/CNNspot_test/test/stylegan2/church_test.csv",
            "/lab/kirito/data/CNNspot_test/test/stylegan2/horse_test.csv",
        ],
    )
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument(
        "--batch_size", type=int, default=128, help="size of the batches"
    )
    parser.add_argument("--num_out_classes", type=int, default=1)
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=[
            "/lab/kirito/clip-fairness/checkpoints/clip-adapter_small_loss_no/progan_lr0.001/clip-adapter_small_loss_no33.pth"
        ],
    )
    parser.add_argument(
        "--model_structure",
        type=str,
        default="clip-fairness_loss",
        help="efficient,ucf_daw",
    )
    parser.add_argument(
        "--backbone", type=str, default="ViT-L/14", help="name of CNN backbone"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="result/no/stylegan2.csv",
        help="file to save the results",
    )

    opt = parser.parse_args()
    print(opt, "!!!!!!!!!!!")

    cuda = True if torch.cuda.is_available() else False

    # prepare the model (detector)
    if opt.model_structure == "clip-fairness_loss":
        model_class = DETECTOR["CLIP_Fairness_loss"]
        # print(model_class)

    if opt.model_structure == "clip-fairness_loss":
        model = model_class(opt)
    if cuda:
        model.to(device)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    output_dir = os.path.dirname(opt.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(opt.output_file, "w", newline="") as csvfile:
        fieldnames = ["Checkpoint", "Test_Path", "Acc", "AP", "AUC", "TPR", "FPR"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for checkpoint in opt.checkpoints:
            ckpt = torch.load(checkpoint, map_location=device)
            model.load_state_dict(ckpt, strict=True)
            print("loading from: ", checkpoint)

            for test_path in opt.test_paths:
                test_dataset = testDataset(test_path, clip_data_transforms["test"])
                test_dataloader = DataLoader(
                    test_dataset, batch_size=opt.batch_size, shuffle=False
                )

                print("%s" % test_path)
                print("-" * 10)
                print("%d batches in total" % len(test_dataloader))

                corrects = 0.0

                pred_label_list = []
                pred_probs_list = []
                label_list = []
                total = 0
                running_corrects = 0
                phase = "val"

                for i, data_dict in enumerate(test_dataloader):
                    bSTime = time.time()
                    phase = "val"
                    model.eval()
                    data, label = data_dict["image"], data_dict["label"]
                    data_dict["image"], data_dict["label"] = data.to(device), label.to(
                        device
                    )
                    data = data.to(device)
                    labels = torch.where(data_dict["label"] != 0, 1, 0)
                    with torch.no_grad():
                        # print(data_dict)
                        preds = model(data_dict, phase)
                        _, preds_label = torch.max(preds["cls"], 1)

                        pred_probs = torch.softmax(preds["cls"], dim=1)[:, 1]
                        total += data_dict["label"].size(0)
                        running_corrects += (
                            (preds_label == data_dict["label"]).sum().item()
                        )
                        p = (preds_label == data_dict["label"]).sum().item()
                        print(p)
                        preds_label = preds_label.cpu().data.numpy().tolist()
                        pred_probs = pred_probs.cpu().data.numpy().tolist()

                    pred_label_list += preds_label
                    pred_probs_list += pred_probs
                    label_list += label.cpu().data.numpy().tolist()
                    bETime = time.time()
                    print(
                        "#{} batch finished, elapsed time: {}".format(
                            i, bETime - bSTime
                        )
                    )

                pred_label_list = np.array(pred_label_list)
                pred_probs_list = np.array(pred_probs_list)
                label_list = np.array(label_list)

                epoch_acc = running_corrects / total

                auc, ap, TPR, FPR, acc = classification_metrics(
                    label_list, pred_probs_list
                )

                print(
                    "Results for {} with checkpoint {}:".format(test_path, checkpoint)
                )
                print(
                    "Acc: {:.4f},  AP: {:.4f},  AUC: {:.4f},  TPR: {:.4f},  FPR: {:.4f}".format(
                        acc, ap, auc, TPR, FPR
                    )
                )
                print()
                print("-" * 10)

                # Save the results to the CSV file
                writer.writerow(
                    {
                        "Checkpoint": checkpoint,
                        "Test_Path": test_path,
                        "Acc": acc,
                        "AP": ap,
                        "AUC": auc,
                        "TPR": TPR,
                        "FPR": FPR,
                    }
                )
