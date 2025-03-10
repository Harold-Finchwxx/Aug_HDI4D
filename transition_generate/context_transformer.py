import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cupy as cp
from tqdm import tqdm
import os
import math

class Encoder(nn.Module):

    def __init__(self, input_shape, device, hidden_dim=512, out_dim=512,  *args, **kwargs):

        """
        args:
        input_shape: input shape tensor in format (batch_size, T, d+1),
        where 'T' is the time span and 'd+1' is dim of context feature plus mask

        device: cuda if available otherwise cpu

        hidden_dim: dim  of hidden layer

        out_dim: dim of the output layer
        """

        super().__init__(*args, **kwargs)

        self.input_shape = input_shape # (batch_size, T, d+1)
        
        self.linear1 = nn.Linear(in_features=self.input_shape[-1], out_features=hidden_dim, device=device)
        self.activation = nn.PReLU
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=out_dim, device=device)

        self.ffn = nn.Sequential(
            self.linear1,
            self.activation,
            self.linear2,
            self.activation
        )

    def forward(self, x):

        return self.ffn(x)


class Decoder(nn.Module):

    """
    general Decoder class, can be turn into instances 
    for transformer decoder and mano decoder and object decoder from TF's output 
    """

    def __init__(self, input_shape, device, hidden_dim=512, out_dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_shape = input_shape # (batch_size, *(T), d)

        self.linear1 = nn.Linear(in_features=self.input_shape[-1], out_features=hidden_dim, device=device)
        self.activation = nn.PReLU
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=out_dim, device=device)

        self.ffn = nn.Sequential(
            self.linear1,
            self.activation,
            self.linear2,
            self.activation
        )

# TODO: modify positionembeding below to apply to mask in formaet [batch, T]
class KPE(nn.Module):

    """
    Keyframe Position Embedding
    """

    def __init__(self, mask, device, hidden_dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # compute the relative position with regard to last context frame
        # and first target frame
        zero_mask = (mask == 0)
        zero_index = torch.nonzero(zero_mask).squeeze()
        mask_start = zero_index[0]
        mask_end = zero_index[-1]
        context_last = mask_start - 1
        target_start = mask_end + 1

        self.relative_position1 = torch.tensor(range(int(mask.shape[1])) - context_last).squeeze
        self.relative_position2 = torch.tensor(range(int(mask.shape[1])) - target_start).squeeze

        self.relative_position = torch.concatenate(
            (self.relative_position1, self.relative_position2),
            dim=0
        )

        self.relative_position = torch.transpose(self.relative_position, 0, 1).to(device)

        # build mapping function
        self.linear1 = nn.Linear(in_features=2, out_features=hidden_dim, device=device)
        self.activattion = nn.PReLU
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=1, device=device)

        self.ffn = nn.Sequential(
            self.linear1,
            self.activattion,
            self.linear2
        )
        
    def forward(self):

        return self.ffn(self.relative_position)


# TODO: modify LRPE below to complete LRPE in the origin paper
class LRPE(nn.Module):

    """
    Learned Relative Position Embedding

    Args:
            T (int): length of window, namely the num of frames in a clip
            dK (int): dim of embedding
            hidden_dim (int): dim of hidden layer
    """

    def __init__(self, T, dk, device, hidden_dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.T = T
        self.dk = dk
        
        # buid mapping between Prel and Erel
        self.linear1 = nn.Linear(in_features=1, out_features=hidden_dim, device=device)
        self.activation = nn.PReLU
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=self.dk, device=device)

        self.ffn = nn.Sequential(
            self.linear1,
            self.activation,
            self.linear2
        )

        # build look-up table
        self.register_buffer('relative_distances', torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(1))

    def forward(self):

        Erel = self.ffn(self.relative_distances)
        return Erel


class ScaleDotProductAttention(nn.Module):
    """

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        
        """
        Input in format [batch_size, head_num, T_length, feature_dim]

        Return value, attention_score
        """

        batch_size, num_head, T_length, dim_f = k.size()

        k_T = k.transpose(-1, -2)
        score = (q @ k_T) / math.sqrt(dim_f)

        # frame mask mechanism
        if mask is not None:
            score.masked_fill(mask == 0, -1e5)

        score = self.softmax(score)
    
        v = score @ v

        return v, score
    

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head:int, dim_f_total:int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_head = n_head
        self.dim_total =dim_f_total
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(in_features=self.dim_total, out_features=self.dim_total)
        self.w_k = nn.Linear(in_features=self.dim_total, out_features=self.dim_total)
        self.w_v = nn.Linear(in_features=self.dim_total, out_features=self.dim_total)
        self.w_conact = nn.Linear(in_features=self.dim_total, out_features=self.dim_total)

        if not self.dim_total % self.n_head == 0:
            raise ValueError("dim_total cannot be divided by num of head evenly")
        
        self.dim_head = self.dim_total // self.n_head


    def forward(self, q, k, v, mask=None):

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        output, attention_map = self.attention(q, k, v, mask=mask)

        output = self.concat(output)
        output = self.w_conact(output)

        return output


    def split(self, x):
        """
        split tensor by number of head

        :param tensor: [batch_size, T_length, d_total]
        :return: [batch_size, num_head, length, d_tensor]
        """
         
        batch_size, num_head, T_length, d_total = x.size()
        tensor = tensor.view(batch_size, T_length, self.n_head, self.dim_head).transpose(1, 2)


    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, num_head, T_length, d_tensor]
        :return: [batch_size, T_length, d_model]
        """
        batch_size, num_head, T_length, d_tensor = tensor.size()
        d_model = num_head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, T_length, d_model)
        return tensor
        


class PositionWiseFeedForwardNetwork(nn.Module):

    def __init__(self, d_total, hidden_dim, device, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear1 = nn.Linear(in_features=d_total, out_features=hidden_dim, device=device)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=d_total, device=device)
        self.activation = nn.PReLU
        self.dropout = nn.Dropout(p=dropout)

        self.pffn = nn.Sequential(
            self.linear1,
            self.activation,
            self.dropout,
            self.linear2
        )

    def forward(self, x):

        return self.pffn(x)
    

class TransformerBlock(nn.Module):

    def __init__(self, num_head, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.multiheadattention = MultiHeadAttention(n_head=num_head, dim_f_total=hidden_dim)
        self.positionwiseffn = PositionWiseFeedForwardNetwork(d_total=hidden_dim, hidden_dim=hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        out = self.multiheadattention(x)
        x = out + x
        x = self.norm1(x)


        out = self.positionwiseffn(x)
        x = out + x
        x = self.norm2(x)
        
        return x
 


class ContextTransformer(nn.Module):


    def __init__(self, batch_size, T, input_dim, num_head, num_blocks, device,
                  hidden_dim=512, out_dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.device = device

        self.encoder = Encoder(input_shape=[batch_size, T, input_dim + 1], device=device)

        self.tfdecoder = Decoder(input_shape=[batch_size, T, hidden_dim], out_dim=out_dim,  device=device)

        #TODO: dim of object marker is to be modified
        self.object_marker_decoder = Decoder(input_shape=[batch_size, T, out_dim], out_dim=12*3, device=device)

        self.left_keypoints_decoder = Decoder(input_shape=[batch_size, T, out_dim], out_dim=21*3, device=device)

        self.right_keypoints_decoder = Decoder(input_shape=[batch_size, T, out_dim], out_dim=21*3, device=device)

        # TODO
        self.multiattention = MultiHeadAttention(n_head=num_head, dim_f_total=hidden_dim)
        self.layernorm = nn.LayerNorm()

        self.pffn = PositionWiseFeedForwardNetwork(d_total=hidden_dim, hidden_dim=hidden_dim)
        #

        self.learned_relative_embedding = LRPE(T=T, dk=input_dim, device=device)


        # build core blocks of trnasformer
        self.tfblocks = nn.Sequential()
        for i in range(num_blocks):
            self.tfblocks.add_module(name=f"block_{i}",
                                      module=TransformerBlock(num_head=num_head, hidden_dim=hidden_dim))

    def forward(self, input, mask):
        
        input[:, (mask == 0), :] = 0

        self.key_frame_embedding = KPE(mask=mask, hidden_dim=self.hidden_dim, device=self.device)
        
        kpe = self.key_frame_embedding.forward()
        x = input + kpe
        x = self.encoder.forward(x)

        # TODO: Add LRPE
        x = self.tfblocks(x)

        y = self.tfdecoder(x)

        object_marker = self.object_marker_decoder(y)
        left_keypoints = self.left_keypoints_decoder(y)
        right_keypoints = self.right_keypoints_decoder(y)

        return object_marker, left_keypoints, right_keypoints
    





        


        
