import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import src.TSALayer
from src.Decoformer import Decoformer
from src.DSW import DSW
from src.HED import Decoder, Encoder
from src.dataloader import DataLoader

import os
import time
import json
import pickle

import warnings

warnings.filterwarnings('ignore')


class Exp_crossformer():
    def __init__(self, args):

    def _build_model(self):
        model = Decoformer(
            self.args.data_dim,
            self.args.in_len,
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model,
            self.args.d_ff,
            self.args.n_heads,
            self.args.e_layers,
            self.args.dropout,
            self.args.baseline,
            self.device
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
        else:
            shuffle_flag = True;
            drop_last = False;
            batch_size = args.batch_size;
        data_set = DataLoader(
            args.data_split,
            args.in_len, args.out_len,
            type=flag
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        print(data_loader)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path + '/' + 'checkpoint.pth')

        return self.model

    def test(self, setting, save_pred=False, inverse=False):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y

    def eval(self, setting, save_pred=False, inverse=False):
        # evaluate a saved model
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len],
            data_split=args.data_split,
            scale=True,
            scale_statistic=args.scale_statistic,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)

        self.model.eval()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return mae, mse, rmse, mape, mspe


parser = argparse.ArgumentParser(description='CrossFormer')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2',
                    help='train/val/test split, can be ratio or number')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')

parser.add_argument('--in_len', type=int, default=96, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=24, help='output MTS length (\tau)')
parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
parser.add_argument('--factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')

parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')
parser.add_argument('--d_model', type=int, default=256, help='dimension of hidden states (d_model)')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers (N)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

parser.add_argument('--baseline', action='store_true',
                    help='whether to use mean of past series as baseline for prediction', default=False)

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')

parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'data_dim': 7, 'split': [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24]},
    'ETTm1': {'data': 'ETTm1.csv', 'data_dim': 7, 'split': [4 * 12 * 30 * 24, 4 * 4 * 30 * 24, 4 * 4 * 30 * 24]},
    'WTH': {'data': 'WTH.csv', 'data_dim': 12, 'split': [28 * 30 * 24, 10 * 30 * 24, 10 * 30 * 24]},
    'ECL': {'data': 'ECL.csv', 'data_dim': 321, 'split': [15 * 30 * 24, 3 * 30 * 24, 4 * 30 * 24]},
    'ILI': {'data': 'national_illness.csv', 'data_dim': 7, 'split': [0.7, 0.1, 0.2]},
    'Traffic': {'data': 'traffic.csv', 'data_dim': 862, 'split': [0.7, 0.1, 0.2]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.data_dim = data_info['data_dim']
    args.data_split = data_info['split']
else:
    args.data_split = string_split(args.data_split)

print('Args in experiment:')
print(args)

Exp = Exp_crossformer

for ii in range(args.itr):
    # setting record of experiments
    setting = 'Crossformer_{}_il{}_ol{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_itr{}'.format(args.data,
                                                                                     args.in_len, args.out_len,
                                                                                     args.seg_len, args.win_size,
                                                                                     args.factor,
                                                                                     args.d_model, args.n_heads,
                                                                                     args.e_layers, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, args.save_pred)