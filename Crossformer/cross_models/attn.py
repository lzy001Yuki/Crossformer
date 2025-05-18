import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from cross_models.cross_embed import DSW_embedding
import numpy as np

from math import sqrt


class Decomposition(nn.Module):
    def __init__(self, win_size):
        super(Decomposition, self).__init__()
        self.win_size = win_size
        # already split the sequence with length 'seq_len'
    def forward(self, x):
        '''
        :param x: batch_size dim time_length
        '''
        mean = nn.AvgPool1d(kernel_size=self.win_size, stride=1, padding=(self.win_size - 1) // 2)
        season = mean(x)
        res = x - season
        return res, season


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channel, win_size):
        super(DLinear, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.decomposition=Decomposition(win_size)
        self.trend_layer = nn.ModuleList()
        self.res_layer = nn.ModuleList()
        self.channel = channel
        for i in range(self.channel):
            self.trend_layer.append(nn.Linear(self.seq_len, self.pred_len))
            self.res_layer.append(nn.Linear(self.seq_len, self.pred_len))
    def forward(self, x):
        batch, seq_len, dim = x.shape
        x = x.transpose(1, 2)
        res, mean = self.decomposition(x)
        res_out = torch.zeros([batch, dim,self.pred_len])
        mean_out = torch.zeros([batch, dim, self.pred_len])
        for i in range(self.channel):
            res_out[:, i, :] = self.res_layer[i](res[:, i, :])
            mean_out[:, i, :] = self.trend_layer[i](mean[:, i, :])
        output = res_out + mean_out
        return output


class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x):
        batch = x.shape[0]
        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(x, 'b ts_d seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out