import numpy as np
import pandas as pd
from Log.print_lib import *


def tracking_target_pars(data_: pd.DataFrame, path_to_dop_materials: str, path_to_save: str, avarage_hours: int = 0) \
        -> None:
    """
    :param data_: Dataframe - данные с омникома
    :param path_to_dop_materials: путь до данных где лежит number_mech_2022 (там информация о контракторе)
    :param path_to_save: путь куда сохранить данных с контракторами и с трекингом
    :param avarage_hours: флаг (1 - нормировать технику по часам сколько в день отработала 0\10\24
    :return: ничего
    """
    whole_target = data_
    # region metch contractors in omnicom data
    # объединение контракторов из Данных из бд трекинга и Данных от коробова(там есть контрактор) а именно из файла
    # number_mech_2022.xlsx

    # path = 'D:/work2/S-krivaia/parsing_exel_Korobov/data/targets_excel/omnicom/omnicom_data.xlsx'
    # data_ = pd.read_excel(path_to_omnic_data)
    # path2 = 'D:/work2/S-krivaia/parsing_exel_Korobov/data/Needed_materials/number_mech_2022.xlsx'
    data_contr = pd.read_excel(path_to_dop_materials + 'number_mech_2022.xlsx')
    whole_target['contractor'] = ''

    unique_number = data_contr['plate_number'].unique()
    for un_num in unique_number:
        data_un_num = data_.loc[data_['vehicle_passport_number'] == un_num]
        if not data_un_num.empty:
            stage = data_un_num.iloc[0, 0]
            d = pd.to_datetime(data_un_num['start_datetime'], format='%Y-%m-%d %H:%M:%S.%f')
            year = d.dt.year
            month = d.dt.month
            day = d.dt.day

            data_un_num.loc[:, data_un_num['year']] = year
            data_un_num.loc[:, data_un_num['month']] = month
            data_un_num.loc[:, data_un_num['day']] = day
            # whole_target = whole_target.append(data_un_num)
            for y in year.unique():
                data_un_num_y = data_un_num.loc[data_un_num['year'] == y]
                if not data_un_num_y.empty:
                    for m in month.unique():
                        data_un_num_m = data_un_num_y.loc[data_un_num_y['month'] == m]
                        if not data_un_num_m.empty:
                            for d in day.unique():
                                data_un_num_d = data_un_num_m.loc[data_un_num_m['day'] == d]
                                if not data_un_num_d.empty:
                                    data_contr_un_num = data_contr.loc[data_contr['plate_number'] == un_num]
                                    whole_target.loc[whole_target.index == data_un_num_d.index[0], 'contractor'] = \
                                        data_contr_un_num['contractor'].unique()[0]

    whole_target['omni'] = whole_target['without_movement'] + whole_target['in_movement']
    whole_target = whole_target.rename(columns={'name_proj': 'stage', 'name_resource': 'res_name'})
    # если контрактор получился пусто, тогда заполнить его 'ООО "СГК-1"'
    whole_target.loc[whole_target['contractor'] == ''] = 'ООО "СГК-1"'

    whole_target.to_excel(path_to_save + 'omnicom+contractors.xlsx')
    print_i(f"SAVED INTERMEDIATE FILE omnicom+contractors.xlsx IN {path_to_save}")
    # whole_target = pd.read_excel('D:\work2\S-krivaia\parsing_exel_Korobov\data\omni_data.xlsx')
    # endregion

    # region calculate target hours
    whole_target_contr = whole_target.loc[whole_target['contractor'] != '']
    whole_target_contr = whole_target_contr.rename(columns={'name': 'stage', 'name.1': 'res_name'})
    unique_res = whole_target_contr['res_name'].unique()
    whole_target_hours = pd.DataFrame(columns=["stage", "contractor", "day", "month", "year", "res_name", "hours_omni"])

    for un_res in unique_res:
        whole_target_contr_un_res = whole_target_contr.loc[whole_target_contr['res_name'] == un_res]
        y = pd.to_datetime(whole_target_contr_un_res['start_datetime'], format='%Y-%m-%d %H:%M:%S.%f').dt.year
        for year in y.unique():
            whole_target_contr_un_y = whole_target_contr_un_res.loc[pd.to_datetime(
                whole_target_contr_un_res['start_datetime'], format='%Y-%m-%d %H:%M:%S.%f').dt.year == year]
            m = pd.to_datetime(whole_target_contr_un_y['start_datetime'], format='%Y-%m-%d %H:%M:%S.%f').dt.month
            for month in m.unique():
                whole_target_contr_un_m = whole_target_contr_un_y.loc[pd.to_datetime(
                    whole_target_contr_un_y['start_datetime'], format='%Y-%m-%d %H:%M:%S.%f').dt.month == month]
                d = pd.to_datetime(whole_target_contr_un_m['start_datetime'], format='%Y-%m-%d %H:%M:%S.%f').dt.day
                for day in d.unique():
                    whole_target_contr_un_d = whole_target_contr_un_m.loc[pd.to_datetime(
                        whole_target_contr_un_m['start_datetime'], format='%Y-%m-%d %H:%M:%S.%f').dt.day == day]
                    s = whole_target_contr_un_d['stage'].unique()
                    for stage in s:
                        whole_target_contr_un_s = whole_target_contr_un_d.loc[whole_target_contr_un_d['stage'] == stage]
                        if avarage_hours:
                            hours_omni = 0
                            for ind, row in whole_target_contr_un_s.iterrows():
                                avg_hours = 24 if (whole_target_contr_un_s['omni'][ind] // 3600) > 10 else 10 if \
                                    (whole_target_contr_un_s['omni'][ind] // 3600) > 5 else 0
                                hours_omni += avg_hours
                        else:
                            hours_omni = whole_target_contr_un_s['omni'].sum()

                        contr = whole_target_contr_un_s['contractor'].unique()[0]
                        whole_target_hours.loc[0] = [stage, contr, day, month, year, un_res, hours_omni]
                        whole_target_hours.index = whole_target_hours.index + 1

    if avarage_hours:
        whole_target_hours.to_excel(path_to_save + "whole_target_hours_omni_10-24.xlsx")
        print_i(f"SAVED FINAL FILE whole_target_hours_omni_10-24.xlsx IN {path_to_save}")
    else:
        whole_target_hours.to_excel(path_to_save + "whole_target_hours_omni.xlsx")
        print_i(f"SAVED FINAL FILE whole_target_hours_omni.xlsx IN {path_to_save}")
    # endregion

