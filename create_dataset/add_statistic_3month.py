from itertools import groupby
from operator import itemgetter, sub
from create_dataset.func_craete_d import *


def add_statistic_100percent(Data: pd.DataFrame) -> pd.DataFrame:
    # print(Data)

    Data_ = Data.copy()
    Data_['month_year'] = 0
    Data_['m_1'] = 0
    Data_['m_2'] = 0
    Data_['m_3'] = 0
    pr_unique = Data['proj_id'].unique()
    con_unique = Data['contr_id'].unique()
    for proj in pr_unique:
        pr_Data = Data.loc[Data['proj_id'] == proj]
        for contr in con_unique:
            pr_con_Data = pr_Data.loc[pr_Data['contr_id'] == contr]
            res_unique = pr_con_Data['res_id'].unique()
            for res in res_unique:
                res_pr_con = pr_con_Data.loc[pr_con_Data['res_id'] == res]
                sort_res_pr_con = res_pr_con.sort_values(['year', 'month'])
                PO = sort_res_pr_con.iloc[:, PO_BEGIN:MONTH_P]
                DF_normlized = (PO - PO.min()) / (PO.max() - PO.min())
                DF_normlized = DF_normlized.replace(np.nan, 0.0)
                sum_rows = DF_normlized.sum(axis=1)
                summa_100 = sum_rows.sum(axis=0)
                month = []
                for ind, m in enumerate(sort_res_pr_con['month']):
                    # if sort_res_pr_con.iloc[ind, YEAR_P] == 1.0:
                    if sort_res_pr_con.iloc[ind, YEAR_P] == 2022.0:
                        m += 12
                    month.append(m)
                sort_res_pr_con['month_year'] = month
                groups = []
                ind_row_2 = []
                ind_row_3 = []
                for k, g in groupby(enumerate(month), lambda x: sub(*x)):
                    items = list(map(itemgetter(1), g))

                    if len(items) == 2:
                        ind_row_2.append(sort_res_pr_con.loc[sort_res_pr_con['month_year'] == items[0]].index[0])
                        ind_row_2.append(sort_res_pr_con.loc[sort_res_pr_con['month_year'] == items[1]].index[0])
                    elif len(items) > 2:
                        k = []
                        for ii in range(len(items)):
                            k.append(sort_res_pr_con.loc[sort_res_pr_con['month_year'] == items[ii]].index[0])
                            ind_row_3.append(sort_res_pr_con.loc[sort_res_pr_con['month_year'] == items[ii]].index[0])
                        groups.append(k)

                sort_res_pr_con['m_1'] = 0.0
                sort_res_pr_con['m_2'] = 0.0
                sort_res_pr_con['m_3'] = 0.0

                for ind_1 in ind_row_2[::2]:
                    for ind_2 in ind_row_2[1::2]:
                        sort_res_pr_con.iloc[sort_res_pr_con.index == ind_2, -3] = sum_rows[ind_1]*100/summa_100

                for i in range(len(groups)):
                    for j in range(len(groups[i])):
                        if j != 0:
                            sort_res_pr_con.iloc[sort_res_pr_con.index == groups[i][j], -3] =\
                                sum_rows[groups[i][j-1]] * 100 / summa_100
                            if j >= 2:
                                sort_res_pr_con.iloc[sort_res_pr_con.index == groups[i][j], -2] = \
                                    sum_rows[groups[i][j - 2]] * 100 / summa_100
                                if j >= 3:
                                    sort_res_pr_con.iloc[sort_res_pr_con.index == groups[i][j], -1] = \
                                        sum_rows[groups[i][j - 3]] * 100 / summa_100

                Data_.iloc[list(sort_res_pr_con.index.values), :] = sort_res_pr_con
    Data_ = Data_.drop('month_year', axis=1)
    Data_ = Data_.set_index('proj_id')
    cols = list(Data_.columns)
    a, b, c, d = cols.index('target'), cols.index('m_1'), cols.index('m_2'), cols.index('m_3')
    cols[a], cols[b], cols[c], cols[d] = cols[b], cols[c], cols[d], cols[a]
    Data_ = Data_[cols]

    return Data_


def add_statistic_previous(Data: pd.DataFrame) -> pd.DataFrame:
    Data_ = Data.copy()
    Data_['month_year'] = 0
    Data_['m_1'] = 0
    Data_['m_2'] = 0
    Data_['m_3'] = 0
    pr_unique = Data['proj_id'].unique()
    con_unique = Data['contr_id'].unique()
    for proj in pr_unique:
        pr_Data = Data.loc[Data['proj_id'] == proj]
        for contr in con_unique:
            pr_con_Data = pr_Data.loc[pr_Data['contr_id'] == contr]
            res_unique = pr_con_Data['res_id'].unique()
            for res in res_unique:
                res_pr_con = pr_con_Data.loc[pr_con_Data['res_id'] == res]
                sort_res_pr_con = res_pr_con.sort_values(['year', 'month'])
                PO = sort_res_pr_con.iloc[:, PO_BEGIN:MONTH_P]
                DF_normlized = (PO - PO.min()) / (PO.max() - PO.min())
                DF_normlized = DF_normlized.replace(np.nan, 0.0)
                sum_rows = DF_normlized.sum(axis=1)
                summa_100 = sum_rows.sum(axis=0)
                month = []
                for ind, m in enumerate(sort_res_pr_con['month']):
                    if sort_res_pr_con.iloc[ind, YEAR_P] == 2022.0:
                        m += 12
                    month.append(m)
                sort_res_pr_con['month_year'] = month
                groups = []
                ind_row_2 = []
                ind_row_3 = []
                for k, g in groupby(enumerate(month), lambda x: sub(*x)):
                    items = list(map(itemgetter(1), g))

                    if len(items) == 2:
                        ind_row_2.append(sort_res_pr_con.loc[sort_res_pr_con['month_year'] == items[0]].index[0])
                        ind_row_2.append(sort_res_pr_con.loc[sort_res_pr_con['month_year'] == items[1]].index[0])
                    elif len(items) > 2:
                        k = []
                        for ii in range(len(items)):
                            k.append(sort_res_pr_con.loc[sort_res_pr_con['month_year'] == items[ii]].index[0])
                            ind_row_3.append(sort_res_pr_con.loc[sort_res_pr_con['month_year'] == items[ii]].index[0])
                        groups.append(k)

                sort_res_pr_con['m_1'] = 0.0
                sort_res_pr_con['m_2'] = 0.0
                sort_res_pr_con['m_3'] = 0.0

                for ind_1 in ind_row_2[::2]:
                    for ind_2 in ind_row_2[1::2]:
                        # sort_res_pr_con.iloc[sort_res_pr_con.index == ind_2, -3] = sum_rows[ind_1]*100/summa_100
                        sort_res_pr_con.iloc[sort_res_pr_con.index == ind_2, -3] = \
                            sum_rows[ind_2]/sum_rows[ind_1]

                for i in range(len(groups)):
                    for j in range(len(groups[i])):
                        if j != 0:
                            # sort_res_pr_con.iloc[sort_res_pr_con.index == groups[i][j], -3] =\
                            #     sum_rows[groups[i][j-1]] * 100 / summa_100
                            if sum_rows[groups[i][j - 1]] != 0:
                                sort_res_pr_con.iloc[sort_res_pr_con.index == groups[i][j], -3] = \
                                    sum_rows[groups[i][j]] / sum_rows[groups[i][j - 1]]
                            if j >= 2:
                                if sum_rows[groups[i][j - 2]] != 0:
                                    sort_res_pr_con.iloc[sort_res_pr_con.index == groups[i][j], -2] = \
                                        sum_rows[groups[i][j]] / sum_rows[groups[i][j - 2]]
                                if j >= 3:
                                    if sum_rows[groups[i][j - 3]] != 0:
                                        sort_res_pr_con.iloc[sort_res_pr_con.index == groups[i][j], -1] = \
                                            sum_rows[groups[i][j]] / sum_rows[groups[i][j - 3]]

                Data_.iloc[list(sort_res_pr_con.index.values), :] = sort_res_pr_con
    Data_ = Data_.drop('month_year', axis=1)
    Data_ = Data_.set_index('proj_id')
    cols = list(Data_.columns)
    a, b, c, d = cols.index('target'), cols.index('m_1'), cols.index('m_2'), cols.index('m_3')
    cols[a], cols[b], cols[c], cols[d] = cols[b], cols[c], cols[d], cols[a]
    Data_ = Data_[cols]

    return Data_
