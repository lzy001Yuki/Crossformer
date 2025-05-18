import torch
import pandas as pd
from DSW import DSW

from torch.utils.data import Dataset, DataLoader

class DataLoader(Dataset):
    def __init__(self, ratio, input_len, output_len, type, data_path='datasets/ETTh1.csv'):
        self.ratio = ratio
        self.data_path = data_path
        self.data = None
        self.time = None
        self.inlen = input_len
        self.outlen = output_len
        self.type = type

    def load_data(self):
        df = pd.read_csv(self.data_path)
        train_num = round(self.ratio[0] * len(df))
        test_num = round(self.ratio[1] * len(df))
        data = df[df.columns[1:]].values
        self.time = df[df.columns[0]].values
        if self.type == 'train':
            self.data = data[:train_num]
        elif self.type == 'test':
            self.data = data[train_num: train_num + test_num]
        else:
            self.data = data[train_num + test_num:]

    def __getitem__(self, index):
        input_ = self.data[index + self.inlen]
        output_ = self.data[index + self.inlen + self.outlen]
        return input_, output_

    # ensure all index selected under 'shuffle' are valid
    def __len__(self):
        return len(self.data) - self.inlen - self.outlen + 1
