import numpy as np
import pandas as pd
from tqdm import tqdm


def prepare_features(df: pd.DataFrame, path: str, save_path: str, stages_dict: dict, tracking: int = 0) -> None:
    nan_res_df = df.loc[np.isnan(df['res_id'])]
    if not nan_res_df.empty:
        df = df.drop(df.index[nan_res_df.index])
    if 'KS6.' in df['PO_id'][0]:
        norm_dict = dict(zip(['KS6.' + str(i) for i in range(1, 374)], range(1, 374)))
    else:
        norm_dict = dict(zip(['PO' + str(i) for i in range(1, 374)], range(1, 374)))

    contr_list = pd.unique(df.contr_id)
    res_list = pd.unique(df.res_id)
    if tracking:
        # month = 24
        day = 24*31
        len_features = 378
    else:
        day = 24  # month за 2 года 12*2
        len_features = 377
    # month_ =
    feature_array = np.zeros((len(contr_list), len(res_list), day, len_features), dtype=float)

    for c in contr_list:
        # ищем позицию контрактора
        contr_position = np.where(contr_list == c)[0][0]

        for y in [2021, 2022]:  # year
            print('year = ', y)

            for m in tqdm(range(1, 13, 1)):  # month
                if tracking:
                    for d in tqdm(range(1, 32, 1)):  # month

                        # находим уников res_id итерируемся по ним:
                        for r in pd.unique(df.loc[(df.contr_id == c) & (df.year == y) & (df.month == m) & (df.day == d),
                                                  'res_id']):

                            # df_res_month = df.loc[(df.contr_id == c) & (df.year == y) &
                            #                       (df.month == m) & (df.res_id == r) &
                            #                       (df.mat_res_name != 'Стоимость') & (pd.notna(df.PO_id)), :]
                            df_res_month = df.loc[(df.contr_id == c) & (df.year == y) &
                                                  (df.month == m) & (df.day == d) & (df.res_id == r) &
                                                  (df.mat_res_name != 'Стоимость') & (pd.notna(df.PO_id)), :]

                            # ищем позицию уника res_id техники
                            try:
                                res_position = np.where(res_list == r)[0][0]
                            except:
                                res_position = np.where(res_list == r)[0]

                            # тут суммируем по нормам:
                            for i in df_res_month.groupby(by=['PO_id']):
                                month_ = (y - 2021) * 12 + m
                                # day_ = (y - 2021) * 12 * m + d
                                day_ = month_ * 31 + d
                                # month = m
                                feature_array[contr_position, res_position, day_, norm_dict[i[0]]] = \
                                    i[1].loc[:, 'act_reg_qty'].sum()

                                # feature_array[contr_position, res_position, month_, norm_dict[i[0]]] = \
                                #     i[1].loc[:, 'act_reg_qty'].sum()


                            sum_feature_array = np.around(feature_array[contr_position, res_position, day_].sum(), 2)
                            sum_dataframe = np.around(df_res_month.act_reg_qty.sum(), 2)

                            # проверка все ли значения сохранили
                            # assert abs(sum_feature_array - sum_dataframe) < 3, f'DEBUG:контрактор_id = {c}, ' \
                            #                                                    f'ресурс_id = {r},' \
                            #                                                    f'суммы часов sum_feature_array = {sum_feature_array},' \
                            #                                                    f'sum_dataframe = {sum_dataframe} должны быть одинаковыми'
                            if y == 2022:
                                # записываем месяц
                                feature_array[contr_position, res_position, day_, -4] = day_ - 31 * month_
                                feature_array[contr_position, res_position, day_, -3] = month_ - 12

                            else:
                                # записываем месяц
                                feature_array[contr_position, res_position, day_, -4] = day_ - 31 * month_
                                feature_array[contr_position, res_position, day_, -3] = month_

                            # записываем год
                            # feature_array[contr_position, res_position, month, -2] = y - 2021
                            feature_array[contr_position, res_position, day_, -2] = y

                            # записываем код техники
                            feature_array[contr_position, res_position, day_, -1] = r

                            # записываем контрактора
                            feature_array[contr_position, res_position, day_, 0] = c
                else:
                    # находим уников res_id итерируемся по ним:
                    for r in pd.unique(
                            df.loc[(df.contr_id == c) & (df.year == y) & (df.month == m), 'res_id']):

                        df_res_month = df.loc[(df.contr_id == c) & (df.year == y) &
                                              (df.month == m) & (df.res_id == r) &
                                              (df.mat_res_name != 'Стоимость') & (pd.notna(df.PO_id)), :]

                        # ищем позицию уника res_id техники
                        res_position = np.where(res_list == r)[0][0]

                        # тут суммируем по нормам:
                        for i in df_res_month.groupby(by=['PO_id']):
                            month_ = (y - 2021) * 12 + m
                            feature_array[contr_position, res_position, month_, norm_dict[i[0]]] = \
                                i[1].loc[:, 'act_reg_qty'].sum()
                        sum_feature_array = np.around(feature_array[contr_position, res_position, month_].sum(), 2)
                        sum_dataframe = np.around(df_res_month.act_reg_qty.sum(), 2)

                        # проверка все ли значения сохранили
                        # assert abs(sum_feature_array - sum_dataframe) < 3, f'DEBUG:контрактор_id = {c}, ' \
                        #                                                    f'ресурс_id = {r},' \
                        #                                                    f'суммы часов sum_feature_array = {sum_feature_array},' \
                        #                                                    f'sum_dataframe = {sum_dataframe} должны быть одинаковыми'
                        if y == 2022:
                            # записываем месяц
                            feature_array[contr_position, res_position, month_, -3] = month_ - 12
                        else:
                            # записываем месяц
                            feature_array[contr_position, res_position, month_, -3] = month_

                        # записываем год
                        # feature_array[contr_position, res_position, month, -2] = y - 2021
                        feature_array[contr_position, res_position, month_, -2] = y

                        # записываем код техники
                        feature_array[contr_position, res_position, month_, -1] = r

                        # записываем контрактора
                        feature_array[contr_position, res_position, month_, 0] = c

        # проверка
        sum_dataframe = np.around(df.loc[(df.contr_id == c) & (df.mat_res_name != 'Стоимость'), 'act_reg_qty'].sum(), 2)
        sum_feature_array = np.around(feature_array[contr_position][:, :, 1:-3].sum(), 2)

        sum_PO_na_values = np.around(
            df.loc[(df.contr_id == c) & (df.mat_res_name != 'Стоимость') & pd.isna(df.PO_id)].act_reg_qty.sum(), 2)
        print(f'INFO: контрактор_id = {c}, сумма значений act_reg_qty без указания норм = {sum_PO_na_values}')
        # проверка все ли значения сохранили
        # assert abs(sum_dataframe - (
        #             sum_feature_array + sum_PO_na_values)) < 1, f'DEBUG:контрактор_id = {c} по всем ресурсам, суммы часов sum_feature_array = {sum_feature_array} ,sum_dataframe = {sum_dataframe} должны быть одинаковыми'
    print('Done')

    # вытягиваем в 2D и убираем нулевые ряды
    f_ar = feature_array.reshape(-1, len_features)
    f_ar = f_ar[np.where(f_ar.sum(axis=1) != 0)[0]]
    # assert f_ar[:, 1:-3].sum() == feature_array[:, :, :, 1:-3].sum()

    name = path.split('/')[-1].replace('.xlsx', '')
    name = name.split('\\')[-1]
    stack = np.full((f_ar.shape[0], 1), stages_dict.get(name))
    f_ar = np.concatenate((stack, f_ar), axis=1)
    if tracking:
        name = 'tracking/'+name

    if not save_path: save_path = './data/prepred_train_data/'

    np.save(f'{save_path}{name}.npy', f_ar, allow_pickle=True)
