import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="gcn_loss")
class GCNLoss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred, f_g, loss_index, label):
        """
        Computes the cross-entropy loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        # Compute the cross-entropy loss
        # print(targets)
        # loss = self.loss_fn(inputs, targets)
        # print(loss, 'cross_entropy loss')
        self.loss1 = self.loss_fn(pred.squeeze(1), label) * 1.0

        f_g_normalized = F.normalize(f_g, p=2, dim=1)
        total_sum = 0.0

        for pair in loss_index:
            i, j = pair
            feature_i = f_g_normalized[i]
            feature_j = f_g_normalized[j]
            dot_product = torch.dot(feature_i, feature_j)
            total_sum += dot_product

        self.loss2 = 1-total_sum / len(loss_index)

        return self.loss1+self.loss2
