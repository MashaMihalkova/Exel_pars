import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from create_dataset.func_craete_d import create_dataset
import os


class PROJDataset(Dataset):
    def __init__(self, proj_data):
        self.X = proj_data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        data = self.X.iloc[ind, :-1]
        target_ = self.X.iloc[ind, -1:]


        if target_ is not None:
            return torch.tensor(data.values), torch.tensor(target_.values)
        else:
            return torch.tensor(data)


class PROJDataset_sequenses(Dataset):
    # TODO :
    #  1 - не верно при репите (один раз повтор по инд, затем опять повтор, но инд прежний, а надо менять)
    #  2 - выбор 4 строк (надо подумать как чтобы выбирать предыдущие 3 а не последующие)
    #  3 - неверная сортировка данных
    #  4 - ошибка при запуске
    def __init__(self, proj_data, mech_res_dict):
        self.X = proj_data
        self.dict_res = {}
        for i in list(mech_res_dict.values()):
            self.dict_res[f'{i}'] = []

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        #
        data = self.X.iloc[ind, :-1]
        print(f'index = {ind}')
        # target_ = self.X.iloc[ind, -1:]
        if str(int(data[-1])) in self.dict_res:
            print(f'res_id = {str(int(data[-1]))} \n value = {self.dict_res[str(int(data[-1]))]}')

            v = self.dict_res[str(int(data[-1]))]
            if v != []:
                ind = v+1
                self.dict_res[str(int(data[-1]))] = v+1
            else:
                self.dict_res[str(int(data[-1]))] = 0
                ind = 0
        data_res = self.X.loc[self.X['res_id'] == data[-1]]
        data_res = data_res.loc[data_res['proj_id'] == data[0]]
        d = data_res.iloc[:, :-1]
        t = data_res.iloc[:, -1:]
        seq = 4
        if d.shape[0] < seq or d.shape[0] < seq+ind:
            shape = d.shape[0]
            for rep in range(seq-shape):
                temp = d.iloc[[ind]]
                temp_t = t.iloc[[ind]]
                d = pd.concat([temp, d])
                t = pd.concat([temp_t, t])
            shape = d.shape[0]
            for k in range(seq+ind - shape):
                temp = d.iloc[[ind]]
                temp_t = t.iloc[[ind]]
                d = pd.concat([temp, d])
                t = pd.concat([temp_t, t])

        if ind <= seq - 1:
            # pad = abs(ind-seq)
            # i_end = ind + seq
            d = d[0:seq]
            # print(f'd shape ={d.shape}')
            t = t[0:seq]
        else:
            # padding = self.X[0].repeat(seq - ind - 1, 1)
            d = d[ind:(ind+seq)]
            print(f'ELSE d shape ={d.shape}')
            # d = torch.cat((padding, d), 0)
            t = t[ind:(ind+seq )]


        # if d.shape[0] < seq:
        #     d = d[:d.shape[0]]
        #     t = t[:d.shape[0]]
        # else:
        #     d = d[:ind+seq]
        #     t = t[:seq]

        if t is not None:
            return torch.tensor(d.values), torch.tensor(t.values)
        else:
            return torch.tensor(data)


