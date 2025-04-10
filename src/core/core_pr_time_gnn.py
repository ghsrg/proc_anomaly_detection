import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def time_loss_fn(pred_time, true_time):
    return F.mse_loss(pred_time.view(-1), true_time.view(-1))


