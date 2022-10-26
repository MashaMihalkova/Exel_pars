import numpy as np
import pandas as pd

PO_BEGIN = 2  # индекс начала РО
LEN_PO = 373  # количество РО
MONTH_P = 375  # индекс начала месяца в project
YEAR_P = 376  # индекс начала года в project
RESOURCE_P = 377  # индекс начала ресурса в project
PROJ_ID = 0  # индекс начала proj_id в project и в target
CONTR_ID = 1  # индекс начала contr_id в project и в target
MONTH_T = 2  # индекс начала месяца в target
YEAR_T = 3  # индекс начала года в target
RESOURCE_T = 4  # индекс начала ресурса в target
VALUE_T = 5  # индекс начала hours в target


def create_dataset(projects, pd_tar):
    dict_data = {'proj_id': [], 'contr_id': []}
    for i in range(LEN_PO):
        dict_data[f'{i}'] = []
    dict_data['month'] = []
    dict_data['year'] = []
    dict_data['res_id'] = []
    dict_data['target'] = []
    dict_pd = {'target': [], 'row': [], 'proj_id': []}
    d = np.load('data_/whole_2021.xlsx')
    for p in projects:
        # if isinstance(p, np.ndarray):
        #     proj = np.load(p)
        # else:
        #     proj = pd.read_excel(p)
        try:
           proj = np.load(p)
        except Exception as e:  # noqa
            proj = pd.read_excel(p)  # noqa
        pd_proj = pd.DataFrame(proj)  # 0/1 - proj/contr, 2-377 - PO
        print(f'proj_name = {p} , proj_id =  {pd_proj[0][0]}')
        # По уникальным контракторам , потом по уникальным месяцам и по годам , потом по ресурсам из ПРОЕКТА
        contr_unoque = pd_proj[:][CONTR_ID].unique()
        month_unique = pd_proj[:][MONTH_P].unique()
        year_unique = pd_proj[:][YEAR_P].unique()
        res_unique = pd_proj[:][RESOURCE_P].unique()
        proj_id = pd_tar.loc[pd_tar[PROJ_ID] == pd_proj[0][0]]

        for contr in contr_unoque:
            contr_id = proj_id.loc[proj_id[CONTR_ID] == contr]
            for month in month_unique:
                month_id = contr_id.loc[contr_id[MONTH_T] == month]
                if month_id.shape[0] != 0:
                    for year in year_unique:
                        if year == 0:
                            y = 2021.0
                        else:
                            y = 2022.0
                        year_id = month_id.loc[month_id[YEAR_T] == y]
                        if year_id.shape[0] != 0:
                            for res in res_unique:
                                res_id = year_id.loc[year_id[RESOURCE_T] == res]
                                if res_id.shape[0] != 0:
                                    target = res_id[VALUE_T].values.sum()
                                    contract_p = pd_proj.loc[pd_proj[CONTR_ID] == contr]
                                    month_p = contract_p.loc[contract_p[MONTH_P] == month]
                                    year_p = month_p.loc[month_p[YEAR_P] == year]
                                    res_p = year_p.loc[year_p[RESOURCE_P] == res]
                                    if res_p.shape[0] != 0:
                                        # print(1)
                                        dict_pd['proj_id'].append(pd_proj[0][0])
                                        dict_pd['target'].append(target)
                                        dict_pd['row'].append(res_p.index[0])

        for i, val in enumerate(dict_pd['row']):
            pd_p = pd_proj.iloc[val]
            dict_data['proj_id'].append(pd_p[PROJ_ID])
            dict_data['contr_id'].append(pd_p[CONTR_ID])
            dict_data['month'].append(pd_p[MONTH_P])
            dict_data['year'].append(pd_p[YEAR_P])
            dict_data['res_id'].append(pd_p[RESOURCE_P])
            for j in range(LEN_PO):
                dict_data[f'{j}'].append(pd_p[j + PO_BEGIN])
            dict_data['target'].append(dict_pd['target'][i])

        dict_pd['proj_id'] = []
        dict_pd['target'] = []
        dict_pd['row'] = []
    return dict_data

#
#
# PO_BEGIN = 2  # индекс начала РО
# LEN_PO = 373  # количество РО
# MONTH_P = 375  # индекс начала месяца в project
# YEAR_P = 376  # индекс начала года в project
# RESOURCE_P = 377  # индекс начала ресурса в project
# PROJ_ID = 0  # индекс начала proj_id в project и в target
# CONTR_ID = 1  # индекс начала contr_id в project и в target
# MONTH_T = 2  # индекс начала месяца в target
# YEAR_T = 3  # индекс начала года в target
# RESOURCE_T = 4  # индекс начала ресурса в target
# VALUE_T = 5  # индекс начала hours в target
#
#
# def create_dataset(projects, pd_tar):
#     dict_data = {'proj_id': [], 'contr_id': []}
#     for i in range(LEN_PO):
#         dict_data[f'{i}'] = []
#     dict_data['month'] = []
#     dict_data['year'] = []
#     dict_data['res_id'] = []
#     dict_data['target'] = []
#     dict_pd = {'target': [], 'row': [], 'proj_id': []}
#
#     for p in projects:
#         try:
#             proj = np.load(p)
#         except Exception as e:  # noqa
#             proj = pd.read_excel(p)  # noqa
#         pd_proj = pd.DataFrame(proj)  # 0/1 - proj/contr, 2-377 - PO
#         print(f'proj_name = {p} , proj_id =  {pd_proj[0][0]}')
#         # По уникальным контракторам , потом по уникальным месяцам и по годам , потом по ресурсам из ПРОЕКТА
#         contr_unoque = pd_proj[:][CONTR_ID].unique()
#         month_unique = pd_proj[:][MONTH_P].unique()
#         year_unique = pd_proj[:][YEAR_P].unique()
#         res_unique = pd_proj[:][RESOURCE_P].unique()
#         proj_id = pd_tar.loc[pd_tar[PROJ_ID] == pd_proj[0][0]]
#
#         for contr in contr_unoque:
#             contr_id = proj_id.loc[proj_id[CONTR_ID] == contr]
#             for month in month_unique:
#                 month_id = contr_id.loc[contr_id[MONTH_T] == month]
#                 if month_id.shape[0] != 0:
#                     for year in year_unique:
#                         if year == 0:
#                             y = 2021.0
#                         else:
#                             y = 2022.0
#                         year_id = month_id.loc[month_id[YEAR_T] == y]
#                         if year_id.shape[0] != 0:
#                             for res in res_unique:
#                                 res_id = year_id.loc[year_id[RESOURCE_T] == res]
#                                 if res_id.shape[0] != 0:
#                                     target = res_id[VALUE_T].values.sum()
#                                     contract_p = pd_proj.loc[pd_proj[CONTR_ID] == contr]
#                                     month_p = contract_p.loc[contract_p[MONTH_P] == month]
#                                     year_p = month_p.loc[month_p[YEAR_P] == year]
#                                     res_p = year_p.loc[year_p[RESOURCE_P] == res]
#                                     if res_p.shape[0] != 0:
#                                         # print(1)
#                                         dict_pd['proj_id'].append(pd_proj[0][0])
#                                         dict_pd['target'].append(target)
#                                         dict_pd['row'].append(res_p.index[0])
#
#         for i, val in enumerate(dict_pd['row']):
#             pd_p = pd_proj.iloc[val]
#             dict_data['proj_id'].append(pd_p[PROJ_ID])
#             dict_data['contr_id'].append(pd_p[CONTR_ID])
#             dict_data['month'].append(pd_p[MONTH_P])
#             dict_data['year'].append(pd_p[YEAR_P])
#             dict_data['res_id'].append(pd_p[RESOURCE_P])
#             for j in range(LEN_PO):
#                 dict_data[f'{j}'].append(pd_p[j + PO_BEGIN])
#             dict_data['target'].append(dict_pd['target'][i])
#
#         dict_pd['proj_id'] = []
#         dict_pd['target'] = []
#         dict_pd['row'] = []
#     return dict_data
