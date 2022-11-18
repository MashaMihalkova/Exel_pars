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

TRACKING_DAY_P = 375  # индекс начала дня в project
TRACKING_MONTH_P = 376  # индекс начала месяца в project
TRACKING_YEAR_P = 377  # индекс начала года в project
TRACKING_RESOURCE_P = 378  # индекс начала ресурса в project
TRACKING_DAY_T = 2   # индекс начала дня в target
TRACKING_MONTH_T = 3  # индекс начала месяца в target
TRACKING_YEAR_T = 4  # индекс начала года в target
TRACKING_RESOURCE_T = 5  # индекс начала ресурса в target
TRACKING_VALUE_T = 6  # индекс начала hours в target


def create_dataset(projects, pd_tar, tracking: int) -> dict:
    dict_data = {'proj_id': [], 'contr_id': []}
    for i in range(LEN_PO):
        dict_data[f'{i}'] = []
    if tracking:
        dict_data['day'] = []
    dict_data['month'] = []
    dict_data['year'] = []
    dict_data['res_id'] = []
    dict_data['target'] = []
    dict_pd = {'target': [], 'row': [], 'proj_id': []}
    # d = np.load('data_/whole_2021.xlsx')
    for p in projects:
        # if isinstance(p, np.ndarray):
        #     proj = np.load(p)
        # else:
        #     proj = pd.read_excel(p)
        try:
           proj = np.load(p, allow_pickle=True)
        except Exception as e:  # noqa
            proj = pd.read_excel(p)  # noqa
        pd_proj = pd.DataFrame(proj)  # 0/1 - proj/contr, 2-377 - PO
        print(f'proj_name = {p} , proj_id =  {pd_proj[1][0]}')
        # По уникальным контракторам , потом по уникальным месяцам и по годам , потом по ресурсам из ПРОЕКТА
        contr_unoque = pd_proj[:][CONTR_ID].unique()
        if tracking:
            month_unique = pd_proj[:][TRACKING_MONTH_P].unique()
            year_unique = pd_proj[:][TRACKING_YEAR_P].unique()
            res_unique = pd_proj[:][TRACKING_RESOURCE_P].unique()
        else:
            month_unique = pd_proj[:][MONTH_P].unique()
            year_unique = pd_proj[:][YEAR_P].unique()
            res_unique = pd_proj[:][RESOURCE_P].unique()
        if not pd.notna(pd_proj[0].unique()[0]):
            proj_id = pd_tar.loc[pd_tar[PROJ_ID] == pd_proj[1][0]]
        else:
            proj_id = pd_tar.loc[pd_tar[PROJ_ID] == pd_proj[0][0]]

        for contr in contr_unoque:
            contr_id = proj_id.loc[proj_id[CONTR_ID] == contr]
            if contr_id.shape[0] != 0:
                for year in year_unique:
                    if year == 2021.0:
                        y = 2021.0
                        # month_n = month
                    else:
                        y = 2022.0
                    if tracking:
                        year_id = contr_id.loc[contr_id[TRACKING_YEAR_T] == y]
                    else:
                        year_id = contr_id.loc[contr_id[YEAR_T] == y]
                    if year_id.shape[0] != 0:
                        for month in month_unique:
                            if month > 12:
                                month_n = month - 12
                            else:
                                month_n = month
                            if tracking:
                                month_id = year_id.loc[year_id[TRACKING_MONTH_T] == month_n]
                            else:
                                month_id = year_id.loc[year_id[MONTH_T] == month_n]

                    # if contr_id.loc[contr_id[YEAR_T]]
                    # if month_id.shape[0] != 0:
                    #     for year in year_unique:
                    #         if year == 0:
                    #             y = 2021.0
                    #             month_n = month
                    #         else:
                    #             y = 2022.0
                    #             if month < 13:
                    #                 month_n = month
                    #             else:
                    #                 month_n = month
                    #         year_id = month_id.loc[month_id[YEAR_T] == y]
                            if month_id.shape[0] != 0:
                                # if year:  # 2022
                                #     month_n = month_n + 12
                                # else:
                                #     month_n = month_n
                                if tracking:
                                    for day in month_id[TRACKING_DAY_T].unique():
                                        day_id = month_id.loc[month_id[TRACKING_DAY_T] == day]
                                        for res in res_unique:
                                            res_id = day_id.loc[day_id[TRACKING_RESOURCE_T] == res]
                                            if res_id.shape[0] != 0:
                                                target = res_id[TRACKING_VALUE_T].values.sum()
                                                contract_p = pd_proj.loc[pd_proj[CONTR_ID] == contr]
                                                month_p = contract_p.loc[contract_p[TRACKING_MONTH_P] == month_n]
                                                day_p = month_p.loc[month_p[TRACKING_DAY_P] == day]
                                                year_p = day_p.loc[day_p[TRACKING_YEAR_P] == year]
                                                res_p = year_p.loc[year_p[TRACKING_RESOURCE_P] == res]
                                                if res_p.shape[0] != 0:
                                                    # print(1)
                                                    dict_pd['proj_id'].append(pd_proj[0][0])
                                                    dict_pd['target'].append(target)
                                                    dict_pd['row'].append(res_p.index[0])
                                else:
                                    for res in res_unique:
                                        res_id = month_id.loc[month_id[RESOURCE_T] == res]
                                        if res_id.shape[0] != 0:
                                            target = res_id[VALUE_T].values.sum()
                                            contract_p = pd_proj.loc[pd_proj[CONTR_ID] == contr]

                                            month_p = contract_p.loc[contract_p[MONTH_P] == month_n]

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
            if tracking:
                dict_data['day'].append(pd_p[TRACKING_DAY_P])
                dict_data['month'].append(pd_p[TRACKING_MONTH_P])
                dict_data['year'].append(pd_p[TRACKING_YEAR_P])
                dict_data['res_id'].append(pd_p[TRACKING_RESOURCE_P])
            else:
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
