import torch
import torch.nn.functional as F


def task_loss_fn(pred_logits, true_labels):
    return F.cross_entropy(pred_logits, true_labels)

