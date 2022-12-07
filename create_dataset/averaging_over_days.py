import pandas as pd


def averaging_over_days(data: pd.DataFrame, N_days: int, path_to_save_dataframe) -> None:
    data_ = data
    data_['date'] = pd.to_datetime(data_[['year', 'month', 'day']])
    df_ = pd.DataFrame()
    for res in data_['res_id'].unique():
        data_res = data_.loc[data_['res_id'] == res]
        for proj in data_res['proj_id'].unique():
            data_res_proj = data_res.loc[data_res['proj_id'] == proj]
            for cont in data_res_proj['contr_id'].unique():
                data_res_proj_cont = data_res_proj.loc[data_res_proj['contr_id'] == cont]
                data_res_proj_cont_sort = data_res_proj_cont.sort_values('date')
                rolling = data_res_proj_cont_sort.rolling(window=N_days)
                rolling_mean = rolling.mean()
                df = rolling_mean.iloc[::N_days, :]
                df = df[df['proj_id'].notna()]
                df_ = pd.concat([df_, df])

    df_.to_excel(path_to_save_dataframe + f'DATA_HOURS_avr_{N_days}_days.xlsx')





    # rolling = data_.rolling(window=N_days)
    # rolling_mean = rolling.mean()
    # df = rolling_mean.iloc[::N_days, :]






