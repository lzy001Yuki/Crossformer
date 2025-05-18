import torch.nn as nn
import torch
from data import Dataset_MTS
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader
import numpy as np
from DSW import DSW
import torch.nn.functional as F
# replace first stage of TSA with DLinear model

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

class Attention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(Attention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / np.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        keys = d_model//n_heads
        out_dim = keys * n_heads
        self.attn = Attention(scale=None, attention_dropout = dropout)
        self.Wq = nn.Linear(d_model, out_dim)
        self.Wk = nn.Linear(d_model, out_dim)
        self.Wv = nn.Linear(d_model, out_dim)
        self.E = nn.Linear(out_dim, d_model)
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        B, L, _ = Q.shape
        _, S, _ = K.shape
        H = self.n_heads

        Q = self.Wq(Q).view(B, L, H, -1)
        K = self.Wk(K).view(B, S, H, -1)
        V = self.Wv(V).view(B, S, H, -1)

        out = self.attention(Q, K, V)

        out = out.view(B, L, -1)

        return self.E(out)

class TwoStageAttention(nn.Module):
    def __init__(self, seq_len, pred_len, channel, win_size, d_model, seg_len, factor, n_heads, dropout=0.1):
        super(TwoStageAttention, self).__init__()
        self.DLinearLayer = DLinear(seq_len, pred_len, channel, win_size)
        self.dsw = DSW(seg_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.seg_len = seg_len
        self.dlinear2cda = nn.Linear(pred_len, seq_len)
        self.norm1 = nn.LayerNorm(seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.router = nn.Parameter(torch.randn(seq_len // seg_len, factor, d_model))
        self.msa1 = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.msa2 = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.MLP = nn.Sequential(nn.Linear(d_model, d_model * 4),
                                nn.GELU(),
                                nn.Linear(d_model * 4, d_model))
    def forward(self, x):
        #DLinear Stage to process Time-relevation
        batch = x.shape[0]
        dlinear_output = self.DLinearLayer(x)
        dim_in = self.norm1(self.dlinear2cda(dlinear_output)).transpose(1, 2)
        dim_embed = self.dsw(dim_in)

        # Cross-Dimension Stage
        dim_send = rearrange(dim_embed, 'b ts_d seg_num d_model -> (b seg_num) ts_d d_model', b = batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat = batch)
        dim_buffer = self.msa1(batch_router, dim_send, dim_send)
        dim_receive = self.msa2(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm2(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP(dim_enc))
        dim_enc = self.norm3(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b = batch)

        return final_out


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x

class GAT_DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, nhid, dropout, alpha, nheads, args):
        super(GAT_DLinear, self).__init__()
        self.gat = GAT(nfeat=num_features, nhid=nhid, nout=num_features, dropout=dropout, alpha=alpha, nheads=nheads)
        self.dlinear = DLinear(seq_len, pred_len, num_features, args.win_size)

    def forward(self, x, adj):
        batch_size, seq_len, num_features = x.size()
        x = x.view(-1, num_features)  # (batch_size * seq_len, num_features)
        adj = adj.repeat(batch_size * seq_len, 1, 1)  # (batch_size * seq_len, num_features, num_features)
        x = self.gat(x, adj)
        x = x.view(batch_size, seq_len, num_features)  # (batch_size, seq_len, num_features)
        x = self.dlinear(x)
        return x



data_set = Dataset_MTS(
            root_path='./datasets/',
            data_path='ETTh1.csv',
            flag='train',
            size=[96,24],
            data_split = [0.7, 0.2, 0.1],
        )

data_loader = DataLoader(
            data_set,
            batch_size=32,
            shuffle=True,drop_last=True)
for i, (batch_x, batch_y) in enumerate(data_loader):
    if i == 0:
        batch, timestep, dim = batch_x.shape
        batch_x = batch_x.float()
        tsa=TwoStageAttention(seq_len=timestep, pred_len=32, channel=dim, win_size=5, d_model=256, n_heads=4, seg_len =6, factor = 10)
        tsa(batch_x)
