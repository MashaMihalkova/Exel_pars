import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
#
# df = pd.read_excel('D:\\work2\\S-krivaia\\data_sila_sibir\\proj_52163_norm_con_correct.xlsx')
# df_targets = pd.read_excel('D:\\work2\\S-krivaia\\data_sila_sibir\\whole_2021_and_2022.xlsx')
#
# contractor = 'ООО "СГК-1"'
# Stage_name = 'КС-3 Амгинская Сила Сибири Этап 5.3'
# df_targets = df_targets.loc[df_targets['stage'] == Stage_name] \
#     .loc[df_targets['contractor'] == contractor]  # \
# # .loc[df_targets['year'] == 2022]
# df_targets.head(3)
#
#
# date_year = df_targets['year']  # pd.DatetimeIndex(df['dt']).year
# date_month = df_targets['month']  # pd.DatetimeIndex(df['dt']).month
# date_y_m = date_year.astype(str) + '-' + date_month.astype(str)
# date_y_m_uniq = list(date_y_m.unique())
#
# resource_name_list = list(pd.unique(df_targets.res_name))
# norms_list = list(pd.unique(df.PO_id))
#
# # resource_list = list(pd.unique(df.id_resource))
# # resource_name_list = list(pd.unique(df.resource_name))
# # norms_list = list(pd.unique(df.PO_id))
#
# # resource_list = list(pd.unique(df.id_resource))
# # date = pd.to_datetime(df['dt'])
# # date_year = pd.DatetimeIndex(df['dt']).year
# # date_month = pd.DatetimeIndex(df['dt']).month
# # date_y_m = date_year.astype(str) + '-' + date_month.astype(str)
# # df['d_y_m'] = date_y_m
# # resource_name_list = list(pd.unique(df['resource_name'].loc[df['d_y_m'] in date_y_m_uniq]))
#
# # date = pd.to_datetime(df['dt'])
# # date_year = pd.DatetimeIndex(df['dt']).year
# # date_month = pd.DatetimeIndex(df['dt']).month
# # date_y_m = date_year.astype(str) + '-' + date_month.astype(str)
# # date_y_m_uniq = list(date_y_m.unique())
# # date = pd.to_datetime(df_targets['dt'])
# date_year = df_targets['year']  # pd.DatetimeIndex(df['dt']).year
# date_month = df_targets['month']  # pd.DatetimeIndex(df['dt']).month
# date_y_m = date_year.astype(str) + '-' + date_month.astype(str)
# date_y_m_uniq = list(date_y_m.unique())
#
# proj_matrix = np.zeros((date_y_m.unique().shape[0], len(resource_name_list), len(norms_list)), dtype=float)
# target_matrix = np.zeros((date_y_m.unique().shape[0], len(resource_name_list)), dtype=float)
#
# for res in resource_name_list:
#     df_res = df.loc[df['resource_name'] == res]
#     # разбиваем по месяцам и годам
#     date_y_m_one = pd.DatetimeIndex(df_res['dt']).year.astype(str) + '-' + pd.DatetimeIndex(df_res['dt']).month.astype(
#         str)
#     df_res['year_month'] = date_y_m_one
#
#     df_res_t = df_targets.loc[df_targets['res_name'] == res]  # & df_targets['contractor'] == 'ООО "СГК-1"']
#     date_y_m_one_t = df_res_t['year'].astype(str) + '-' + df_res_t['month'].astype(str)
#     df_res_t['year_month'] = date_y_m_one_t
#     for d_y_m_res in date_y_m_one.unique():
#         date_y_m_res = df_res.loc[df_res['year_month'] == d_y_m_res]
#         date_y_m_res_t = df_res_t.loc[df_res_t['year_month'] == d_y_m_res]
#
#         if date_y_m_res_t.shape[0] != 0:
#             target_matrix[date_y_m_uniq.index(d_y_m_res), resource_name_list.index(res)] = date_y_m_res_t['hours'].values[0]
#         # разбиваем по нормам и суммируем сумму заносим в матрицу в определенное место по дате ресурсу и по нормам
#             for norm in date_y_m_res['PO_id'].unique():
#                 norm_y_m_res = date_y_m_res.loc[date_y_m_res['PO_id'] == norm]
#                 sum_act = norm_y_m_res['act_reg_qty'].sum()
#                 proj_matrix[date_y_m_uniq.index(d_y_m_res), resource_name_list.index(res), norms_list.index(norm)] = sum_act
#
# # каждой строчки по ресурсу и по дате proj_matrix[0, 0,:] соответствует 1 таргет
# # надо таргет тоже оформить так : target_matrix[0, 0] дата, ресурс
#
# print(proj_matrix)
# print(target_matrix)
# print(1)
# dict_data = {'date':[], 'resource':[],'target':[]}
# for i in range(proj_matrix.shape[2]):
#     dict_data[f'{norms_list[i]}'] = []
# # resource_list
# # создать матрицу куда внести все значения и в размерность 2 data и resource
# for month in range(target_matrix.shape[0]):
#     for res in range(target_matrix.shape[1]):
#         if target_matrix[month][res] != 0.0:
#             dict_data['date'].append(date_y_m_uniq[month])
#             dict_data['resource'].append(resource_name_list[res])
#             dict_data['target'].append(target_matrix[month][res])
#             for i in range(proj_matrix.shape[2]):
#                 dict_data[f'{norms_list[i]}'].append(proj_matrix[month][res][i])
#
# print(dict_data)
# pd_dataframe = pd.DataFrame(dict_data)
# print(pd_dataframe)


targets = np.load('data/target_array.npy')
proj = np.load('data/КС-3 Амгинская Сила Сибири Этап 5.3.npy')

print(1)
pd_tar = pd.DataFrame(targets) # 0/1-contr/proj, 2-month, 3- year, 4 - , 5 -
pd_proj = pd.DataFrame(proj) # 0/1 - proj/contr, 2-377 - PO


class PROJDataset(Dataset):
    def __init__(self, proj_data, targets=None):
        self.X = proj_data
        self.y = targets

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):

        data = self.X.iloc[i, :]
        proj_id = self.X.loc[i][0]
        contract_id = self.X.loc[i][1]
        month = self.X.loc[i][375]
        year = self.X.loc[i][376]
        res = self.X.loc[i][377]
        # t_proj_id = self.y.loc[i][0]
        # t_contract_id = self.y.loc[i][1]
        # t_month = self.y.loc[i][2]
        # t_year = self.y.loc[i][3]
        # t_res = self.y.loc[i][4]
        # 2021 - 0, 2022 - 1

        target = None
        proj_y = self.y[self.y.loc[:][0] == proj_id]
        cont_Y = proj_y[proj_y.loc[:][1] == contract_id]
        month_y = cont_Y[cont_Y.loc[:][2] == month]
        # for row in range(self.y.shape[0]):
        for row,d in enumerate(month_y.index):
            # if self.y.loc[row][0] == proj_id:
            #     if self.y.loc[row][1] == contract_id:
            #         if self.y.loc[row][2] == month:
                        if year == 0:
                            y = 2021
                        else:
                            y = 2022
                        if int(self.y.loc[d][3]) == y:
                            if self.y.loc[d][4] == res:
                                target = self.y.loc[d][5]
                                continue


        if target is not None:
            return (torch.tensor(data.values), torch.tensor(target.values))
        else:
            next(i)
            # return torch.tensor(data)


train_dataset = PROJDataset(pd_proj, pd_tar)

# train_dataset = TimeseriesDataset(X_lstm, y_lstm, seq_len=4)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 3, shuffle = False)

for i, d in enumerate(train_loader):
    print(i, d[0].shape, d[1].shape)

print(1)

