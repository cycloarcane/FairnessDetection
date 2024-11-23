import sys

from detectors import DETECTOR
import torch
import math
import random

from torch.optim import lr_scheduler
import numpy as np
import os
import os.path as osp
import copy
from utils1 import Logger

import torch.backends.cudnn as cudnn
# from dataset.pair_dataset import pairDataset
# from dataset.third_dataset_aug import thirdAugDataset
from dataset.mydataset_train import thirdAugDataset
from dataset.mydataset_val import valDataset
import csv
import argparse

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser("Center Loss Example")
parser.add_argument('--lamda1', type=float, default=0.1,
                    help="alpha_i in daw-fdd, (0.0~1.0)")
parser.add_argument('--lamda2', type=float, default=0.01,
                    help="alpha in daw-fdd,(0.0~1.0)")
parser.add_argument('--weight', type=float, default=0.5,
                    help="alpha in dag-fdd, (0.0~1.0)")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lrf', type=float, default=0.01)
parser.add_argument('--batchsize', type=int, default=2, help=".")
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--dataname', type=str, default='progan',
                    help='ff++, celebdf, dfd, dfdc')
parser.add_argument('--task', type=str, default='clip-adapter_2loss',
                    help='ori, dag-fdd, daw-fdd')
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--train_datapath', type=str,
                    default='/lab/kirito/data/CNNspot_test/train/progan/datacsv')
parser.add_argument('--val_datapath', type=str,
                    default='/lab/kirito/data/CNNspot_test/val/val.csv')
parser.add_argument("--continue_train", action='store_true')
parser.add_argument("--checkpoints", type=str, default='',
                    help="continue train model path")
parser.add_argument("--model", type=str, default='clip-adapter_small',
                    help="model structure")
parser.add_argument("--backbone", type=str, default="ViT-L/14", help="name of CNN backbone")

args = parser.parse_args()

###### different data transform for different backbone #######
# if args.model == 'XceptionRGB':
#     from transform import xception_default256_data_transforms as data_transforms


###### load data ######
train_dataset = thirdAugDataset(args.train_datapath)
c = train_dataset[0]
val_dataset = valDataset(args.val_datapath)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize,
                                                collate_fn=thirdAugDataset.collate_fn)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=40, collate_fn=valDataset.collate_fn)

train_dataset_sizes = len(train_dataset)
val_dataset_sizes = len(val_dataset)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# prepare the model (detector)
model_class = DETECTOR['CLIP_Adapter']


def classification_metrics(label, prediction):
    auc = roc_auc_score(label, prediction)
    CM = confusion_matrix(label, prediction >= 0.5)
    acc = accuracy_score(label, prediction >= 0.5)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    return auc, TPR, FPR, acc


##### calcuate ffpr score calculation during val ################


def cal_ffpr_score(preds, labels, intersec_label):
    efpr_s = 0.0

    logits_idx = (labels < 0.5)
    if np.sum(logits_idx) > 0:
        allg = np.sum(preds[logits_idx] == 1) / np.sum(logits_idx)
    else:
        allg = 0.0
        print("no real data in this batch")

    for j in list(np.unique(intersec_label)):
        groupa = 0.0
        groupb = np.sum(labels[intersec_label == j] < 0.5)

        if groupb != 0:
            groupa = np.sum(preds[(intersec_label == j) & (logits_idx)] == 1)
            group = groupa / groupb
        else:
            group = 0.0

        efpr_s += np.abs(group - allg)

    return efpr_s


##### calcuate feo score during val#############


def cal_feo_score(preds, labels, intersec_label):
    eo_score_r = 0.0
    eo_score_f = 0.0

    logits_idx_r = (labels < 0.5)
    if np.sum(logits_idx_r) > 0:
        allg_r = np.sum(preds[logits_idx_r] == 1) / np.sum(logits_idx_r)
    else:
        allg_r = 0.0
        print("no real data in this batch")

    for j in range(8):
        groupa_r = 0.0
        groupb_r = np.sum(labels[intersec_label == j] < 0.5)

        if groupb_r != 0:
            groupa_r = np.sum(
                preds[(intersec_label == j) & (logits_idx_r)] == 1)
            group_r = groupa_r / groupb_r
        else:
            group_r = 0.0

        eo_score_r += np.abs(group_r - allg_r)

    logits_idx_f = (labels >= 0.5)
    if np.sum(logits_idx_f) > 0:
        allg_f = np.sum(preds[logits_idx_f] == 1) / np.sum(logits_idx_f)
    else:
        allg_f = 0.0
        print("no real data in this batch")

    for j in range(8):
        groupa_f = 0.0
        groupb_f = np.sum(labels[intersec_label == j] >= 0.5)

        if groupb_f != 0:
            groupa_f = np.sum(
                preds[(intersec_label == j) & (logits_idx_f)] == 1)
            group_f = groupa_f / groupb_f
        else:
            group_f = 0.0

        eo_score_f += np.abs(group_f - allg_f)

    return (eo_score_r + eo_score_f)


###### calculate G_auc during val ##############


def auc_gap(preds, labels, intersec_label):
    auc_all_sec = []

    for j in list(np.unique(intersec_label)):
        pred_section = preds[intersec_label == j]
        labels_section = labels[intersec_label == j]
        try:
            auc_section, _, _, _ = classification_metrics(
                labels_section, pred_section)
            auc_all_sec.append(auc_section)
        except:
            continue
    return max(auc_all_sec) - min(auc_all_sec)


def cal_foae_score(preds, labels, intersec_label):
    acc_all_sec = []

    for j in list(np.unique(intersec_label)):
        pred_section = preds[intersec_label == j]
        labels_section = labels[intersec_label == j]
        try:
            _, _, _, acc_section = classification_metrics(
                labels_section, pred_section)
            acc_all_sec.append(acc_section)
        except:
            continue
    return max(acc_all_sec) - min(acc_all_sec)


# train and evaluation


def train(model, optimizer, num_epochs, start_epoch):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        phase = 'train'
        model.train()

        total_loss = 0.0

        for idx, data_dict in enumerate(train_data_loader):

            real_imgs, real_labels, fake_imgs, fake_labels = data_dict['real_images'], data_dict['real_labels'], \
                data_dict['fake_images'], data_dict['fake_labels']

            real_labels = real_labels[:len(real_labels) // real_imgs.shape[0]]
            fake_labels = fake_labels[:len(fake_labels) // real_imgs.shape[0]]
            '''print(real_labels.shape)
            print(fake_labels.shape)'''

            # with torch.set_grad_enabled(phase == 'train'):

            for i in range(20):
                convert = random.randint(0, 1)
                a_fakeimg = []
                b_realimg = []
                prompt_fin = []

                for batch in range(real_imgs.shape[0]):
                    real_img_tmp = real_imgs[batch]
                    fake_img_tmp = fake_imgs[batch]

                    fake_img = fake_img_tmp[i]
                    fake_img = fake_img.repeat(19, 1, 1, 1, )

                    reala_img = real_img_tmp[i]
                    reala_img = reala_img.repeat(19, 1, 1, 1, )

                    if convert == 0:
                        fakea_img = torch.cat((fake_img, reala_img), dim=0)
                    elif convert == 1:
                        fakea_img = torch.cat((reala_img, fake_img), dim=0)

                    real_img = torch.cat((real_img_tmp[:i], real_img_tmp[i + 1:]))
                    realb_img = torch.cat((real_img, real_img), dim=0)

                    real_label = torch.cat((real_labels[:i], real_labels[i + 1:])).squeeze()
                    fake_label = torch.cat((fake_labels[:i], fake_labels[i + 1:])).squeeze()

                    if convert == 0:
                        prompt_tmp = torch.cat((fake_label, real_label), dim=0)
                    elif convert == 1:
                        prompt_tmp = torch.cat((real_label, fake_label), dim=0)

                    '''print(fakea_img.shape)
                    print(realb_img.shape)
                    print(prompt_tmp)'''

                    a_fakeimg.append(fakea_img)
                    b_realimg.append(realb_img)
                    prompt_fin.append(prompt_tmp)

                fakea_img = torch.cat(a_fakeimg, dim=0)
                realb_img = torch.cat(b_realimg, dim=0)
                prompt = torch.cat(prompt_fin, dim=0)
                '''print(fakea_img.shape)
                print(realb_img.shape)
                print(prompt)'''

                data_dict['real_images'], data_dict['real_labels'], data_dict['fake_images'], data_dict[
                    'fake_labels'] = realb_img.to(device), real_label.to(device), fakea_img.to(device), prompt.to(
                    device)

                preds1, preds2 = model(data_dict, phase)
                losses = model.get_losses(data_dict, preds1, preds2)
                firstloss = losses['firstloss']
                finloss = losses['finloss']
                finloss = firstloss + finloss
                optimizer.zero_grad()
                finloss.backward()
                optimizer.step()
                # print('image {}:'.format(i), format(firstloss.item(), ".4e"), '   ', format(finloss.item(), ".4e"))
            if idx % 1 == 0:
                # compute training metric for each batch data
                batch_metrics = model.get_train_metrics(data_dict, preds2)
                print('#{} batch_metric{}'.format(idx, batch_metrics))

            total_loss += finloss.item() * real_imgs.size(0)

        epoch_loss = total_loss / train_dataset_sizes
        print('Epoch: {} Loss: {:.4f}'.format(epoch, epoch_loss))

        # evaluation
        # if (epoch+1) % 5 == 0:
        if (epoch + 1) % 1 == 0:

            savepath = 'checkpoints/' + args.model + '/' + args.dataname + '_lr' + str(args.lr)

            temp_model = savepath + "/" + args.model + str(epoch) + '.pth'
            torch.save(model.state_dict(), temp_model)

            print()
            print('-' * 10)

            phase = 'val'
            model.eval()
            running_corrects = 0
            total = 0

            pred_label_list = []
            pred_probs_list = []
            label_list = []

            for idx, data_dict in enumerate(val_data_loader):
                imgs, labels = data_dict['image'], data_dict['label']

                labels = torch.where(data_dict['label'] != 0, 1, 0)

                data_dict['image'], data_dict['label'] = imgs.to(device), labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(data_dict, phase)
                    _, preds_label = torch.max(preds['cls'], 1)
                    pred_probs = torch.softmax(
                        preds['cls'], dim=1)[:, 1]
                    total += data_dict['label'].size(0)
                    running_corrects += (preds_label ==
                                         data_dict['label']).sum().item()

                    preds_label = preds_label.cpu().data.numpy().tolist()
                    pred_probs = pred_probs.cpu().data.numpy().tolist()
                # losses = model.get_losses(data_dict, preds)
                pred_label_list += preds_label
                pred_probs_list += pred_probs
                label_list += labels.cpu().data.numpy().tolist()
                if idx % 50 == 0:
                    batch_metrics = model.get_test_metrics()

                    print('#{} batch_metric{{"acc": {}, "auc": {}, "ap": {}}}'.format(idx,
                                                                                      batch_metrics['acc'],
                                                                                      batch_metrics['auc'],
                                                                                      batch_metrics['ap']))

            pred_label_list = np.array(pred_label_list)

            pred_probs_list = np.array(pred_probs_list)
            label_list = np.array(label_list)

            epoch_acc = running_corrects / total

            auc, TPR, FPR, _ = classification_metrics(
                label_list, pred_probs_list)

            print('Epoch {} Acc: {:.4f}  auc: {}, tpr: {}, fpr: {}'.format(
                epoch, epoch_acc, auc, TPR, FPR))
            with open(savepath + "/val_metrics.csv", 'a', newline='') as csvfile:
                columnname = ['epoch', 'epoch_acc', 'AUC all', 'TPR all', 'FPR all']
                writer = csv.DictWriter(csvfile, fieldnames=columnname)
                writer.writerow(
                    {'epoch': str(epoch), 'epoch_acc': str(epoch_acc), 'AUC all': str(auc), 'TPR all': str(TPR),
                     'FPR all': str(FPR)})

            print()
            print('-' * 10)

    return model, epoch


def main():
    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    '''if args.task == 'XceptionRGB_spe_aug':
        sys.stdout = Logger(osp.join(
            './Final_Misleading/checkpoints/' + args.model + '/' + args.dataname + '_' + args.task + '/lamda1_' + str(
                args.lamda1) + '_lamda2_' + str(args.lamda2) + '_lr' + str(args.lr) + '/log_training.txt'))'''
    sys.stdout = Logger(osp.join(
        'checkpoints/' + args.model + '/' + args.dataname + '_lr' + str(args.lr) + '/log_training.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    model = model_class(args)
    model.to(device)

    # weights_part1 = torch.load("/home/ubuntu/shahur/Misleading/checkpoints/paper_metric/XceptionMis_Newmethod_newscam/ff++_XceptionMis_Newmethod_newscam_fixsrm_noPretrainedSRM_twostreams_newStrategy/lamda1_0.1_lamda2_0.01_lr0.001/XceptionMis_Newmethod_newscam0.pth")
    # model_dict = model.state_dict()
    # # 创建一个新的字典pretrained_dict，只包含weights_part1中存在于model_dict中的键值对
    # pretrained_dict = {k: v for k, v in weights_part1.items() if k in model_dict}

    # for k, v in weights_part1.items():
    #     if k in model_dict:
    #         print(k)
    # 更新现有的model_dict
    # model_dict.update(pretrained_dict)
    # # 加载我们真正需要的state_dict
    # model.load_state_dict(model_dict)
    for name, para in model.named_parameters():
        # if para.requires_grad:
        #  print(name)
        if "text_encoder" in name or "image_encoder" in name:
        # if "image_encoder" in name:
            para.requires_grad_(False)
        '''if para.requires_grad:
         print(name)'''
    #     if "conv3" not in name and "bn3" not in name and "conv4" not in name and "bn4" not in name and "last_linear" not in name and "adjust_channel" not in name:
    #         para.requires_grad_(False)
    #     else:
    #         print("training {}".format(name))

    # model_ft = nn.DataParallel(model_ft, device_ids=[0, 1, 2, 3]).cuda()

    start_epoch = 0

    if args.continue_train and args.checkpoints != '':
        state_dict = torch.load(args.checkpoints)
        model.load_state_dict(state_dict)
        print('continue train from: ', args.checkpoints)
        start_epoch = int(
            ((args.checkpoints).split('/')[-1]).split('.')[0][8:]) + 1

    # optimize
    # optimizer4nn = optim.SGD(params_to_update, lr=args.lr,
    #                          momentum=0.9, weight_decay=5e-03)

    # optimizer = optimizer4nn
    # change to SAM optimizer
    # define an optimizer for the "sharpness-aware" update
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.0002,
        weight_decay=0.0005,
        betas=(0.9, 0.999),
        eps=0.00000001,
        amsgrad='false',
    )

    # model_ft, epoch = train(model_ft, criterion, optimizer,
    #                         exp_lr_scheduler, num_epochs=200, start_epoch=start_epoch)
    model, epoch = train(model, optimizer, num_epochs=args.epochs, start_epoch=start_epoch)

    if epoch == args.epochs - 1:
        print("training finished!")
        exit()


if __name__ == '__main__':
    main()
