import numpy as np
import pandas as pd

df = pd.read_excel('D:\\work2\\S-krivaia\\data_sila_sibir\\proj_52163_norm_con_correct.xlsx')
df_targets = pd.read_excel('D:\\work2\\S-krivaia\\data_sila_sibir\\whole_2021_and_2022.xlsx')

contractor = 'ООО "СГК-1"'
Stage_name = 'КС-3 Амгинская Сила Сибири Этап 5.3'
df_targets = df_targets.loc[df_targets['stage'] == Stage_name] \
    .loc[df_targets['contractor'] == contractor]  # \
# .loc[df_targets['year'] == 2022]
df_targets.head(3)


date_year = df_targets['year']  # pd.DatetimeIndex(df['dt']).year
date_month = df_targets['month']  # pd.DatetimeIndex(df['dt']).month
date_y_m = date_year.astype(str) + '-' + date_month.astype(str)
date_y_m_uniq = list(date_y_m.unique())

resource_name_list = list(pd.unique(df_targets.res_name))
norms_list = list(pd.unique(df.PO_id))

# resource_list = list(pd.unique(df.id_resource))
# resource_name_list = list(pd.unique(df.resource_name))
# norms_list = list(pd.unique(df.PO_id))

# resource_list = list(pd.unique(df.id_resource))
# date = pd.to_datetime(df['dt'])
# date_year = pd.DatetimeIndex(df['dt']).year
# date_month = pd.DatetimeIndex(df['dt']).month
# date_y_m = date_year.astype(str) + '-' + date_month.astype(str)
# df['d_y_m'] = date_y_m
# resource_name_list = list(pd.unique(df['resource_name'].loc[df['d_y_m'] in date_y_m_uniq]))

# date = pd.to_datetime(df['dt'])
# date_year = pd.DatetimeIndex(df['dt']).year
# date_month = pd.DatetimeIndex(df['dt']).month
# date_y_m = date_year.astype(str) + '-' + date_month.astype(str)
# date_y_m_uniq = list(date_y_m.unique())
# date = pd.to_datetime(df_targets['dt'])
date_year = df_targets['year']  # pd.DatetimeIndex(df['dt']).year
date_month = df_targets['month']  # pd.DatetimeIndex(df['dt']).month
date_y_m = date_year.astype(str) + '-' + date_month.astype(str)
date_y_m_uniq = list(date_y_m.unique())

proj_matrix = np.zeros((date_y_m.unique().shape[0], len(resource_name_list), len(norms_list)), dtype=float)
target_matrix = np.zeros((date_y_m.unique().shape[0], len(resource_name_list)), dtype=float)

for res in resource_name_list:
    df_res = df.loc[df['resource_name'] == res]
    # разбиваем по месяцам и годам
    date_y_m_one = pd.DatetimeIndex(df_res['dt']).year.astype(str) + '-' + pd.DatetimeIndex(df_res['dt']).month.astype(
        str)
    df_res['year_month'] = date_y_m_one

    df_res_t = df_targets.loc[df_targets['res_name'] == res]  # & df_targets['contractor'] == 'ООО "СГК-1"']
    date_y_m_one_t = df_res_t['year'].astype(str) + '-' + df_res_t['month'].astype(str)
    df_res_t['year_month'] = date_y_m_one_t
    for d_y_m_res in date_y_m_one.unique():
        date_y_m_res = df_res.loc[df_res['year_month'] == d_y_m_res]
        date_y_m_res_t = df_res_t.loc[df_res_t['year_month'] == d_y_m_res]

        if date_y_m_res_t.shape[0] != 0:
            target_matrix[date_y_m_uniq.index(d_y_m_res), resource_name_list.index(res)] = date_y_m_res_t['hours'].values[0]
        # разбиваем по нормам и суммируем сумму заносим в матрицу в определенное место по дате ресурсу и по нормам
            for norm in date_y_m_res['PO_id'].unique():
                norm_y_m_res = date_y_m_res.loc[date_y_m_res['PO_id'] == norm]
                sum_act = norm_y_m_res['act_reg_qty'].sum()
                proj_matrix[date_y_m_uniq.index(d_y_m_res), resource_name_list.index(res), norms_list.index(norm)] = sum_act

# каждой строчки по ресурсу и по дате proj_matrix[0, 0,:] соответствует 1 таргет
# надо таргет тоже оформить так : target_matrix[0, 0] дата, ресурс

print(proj_matrix)
print(target_matrix)
print(1)
# resource_list
#  создать матрицу куда внести все значения