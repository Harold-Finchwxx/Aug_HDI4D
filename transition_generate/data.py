import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cupy as cp
from tqdm import tqdm
import os
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pickle


class HDI4DDataset(Dataset):

    """
    """

    def __init__(self, data_dir, T_window=60, transit_length=30):
        super().__init__()


        self.data_dir = data_dir
        self.T_window = T_window
        self.transit_length = transit_length

        self.object_marker_path = os.path.join(data_dir, "object_marker.pickle")
        self.left_keypoints_path = os.path.join(data_dir, "hand_anno_vit/cluster/right/keypoints.pkl")
        self.right_keypoints_path = os.path.join(data_dir, "hand_anno_vit/cluster/left/keypoints.pkl")

        with open(self.object_marker_path, 'rb') as f:
            self.object_marker = pickle.load(f)

        with open(self.left_keypoints_path, 'rb') as f:
            self.left_keypoints = pickle.load(f)

        with open(self.right_keypoints_path, 'rb') as f:
            self.right_keypoints = pickle.load(f)

    def __len__(self):

        max_sample = len(self.object_marker) - 1 - self.T_window
        return max_sample
    
    def __getitem__(self, index):

        object_marker = torch.tensor(self.object_marker[index : (index + self.T_window)][0:12])
        left_keypoints = torch.tensor(self.left_keypoints[index : (index + self.T_window)])
        right_keypoints = torch.tensor(self.right_keypoints[index : (index + self.T_window)])

        mask = np.ones(self.T_window)
        mask[(self.T_window - self.transit_length - 5 - 1) : (self.T_window - 5)] = 0
        mask = torch.tensor(mask)

        feature_concat = torch.cat((object_marker, left_keypoints, right_keypoints),
                            dim=1).squeeze


        return feature_concat, mask




