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
from data import HDI4DDataset
import context_transformer
from context_transformer import ContextTransformer
import torch.optim as optim



def train_context_tf(num_epoch=2000, batch_size=16, num_workers=4, num_head=8, 
                     num_block=6, hidden_dim=512, tf_out_dim=512, lr=1e-4,
                     datasets_dir="HDI4D"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_path = os.listdir(datasets_dir)[1:5]
    datasets_path.append(os.listdir(datasets_dir)[6])

    train_datasets = []
    test_datasets = []
    train_loaders = []
    test_loaders = []

    for dataset_path in datasets_path:

        full_datsset = HDI4DDataset(data_dir=dataset_path)

        train_size = int(len(full_datsset) * 0.8)
        test_size = len(full_datsset) - train_size

        train_dataset, test_dataset = random_split(dataset=full_datsset, lengths=[train_size, test_size])

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True) 
        # TODO: try and test if pin_memeory and num-workers are well set

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        
    
    train_model = ContextTransformer(batch_size=batch_size,
                                     T=60,
                                     input_dim=(12 + 21 +21) * 3,
                                     num_head=num_head,
                                     num_blocks=num_block,
                                     device=device,
                                     hidden_dim=hidden_dim,
                                     out_dim=tf_out_dim).to(device=device)
    
    optimizer = optim.Adam(train_model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epoch), desc="Training"):



        


