import torch.nn as nn
from einops import rearrange
import torch
from torch.utils.data import Dataset, DataLoader



class DSW(nn.Module):
    def __init__(self, seg_len, d):
        super(DSW, self).__init__()
        self.seg_len = seg_len
        self.d = d
        self.linear = nn.Linear(self.seg_len, self.d)

    def forward(self, x):
        batch, timestep, dim = x.size()
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=dim)

        return x_embed




