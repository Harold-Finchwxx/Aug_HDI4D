import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cupy as cp
from tqdm import tqdm
import os
import math


def L2_position_loss(predict, gt, mask):

    """
    tensor in shape[batch, T, points, coordinate(x,y,z)]
    """
    loss = torch.square(predict[mask] - gt[mask])
    loss = torch.sum(loss, dim=-1)
    loss = torch.sqrt(loss)
    loss = torch.sum(loss, dim=-1)
    loss = torch.mean(loss)

    loss = torch.mean(loss)

    return loss
        

def L1_position_loss(predict, gt, mask):

    """
    tensor in shape[batch, T, points, coordinate(x,y,z)]
    """
    loss = torch.abs(predict[mask] - gt[mask])
    loss = torch.sum(loss, dim=-1)
    
    loss = torch.sum(loss, dim=-1)
    loss = torch.mean(loss)

    loss = torch.mean(loss)

    return loss


def L2_smooth_loss(predict, gt, mask):

    """
    tensor in shape[batch, T, points, coordinate(x,y,z)]
    """
    loss = torch.abs(predict[mask+1] - predict[mask])
    loss = torch.sum(loss, dim=-1)
    loss = torch.sqrt(loss)
    loss = torch.sum(loss, dim=-1)
    loss = torch.mean(loss)

    loss = torch.mean(loss)

    return loss