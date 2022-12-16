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
                month_av = data_res_proj_cont_sort.loc[:, 'month']


                rolling = data_res_proj_cont_sort.rolling(window=N_days)
                rolling_mean = rolling.mean()
                rolling_max = rolling.max()
                df = rolling_mean.iloc[::N_days, rolling_mean.columns != 'month']
                df_m = rolling_max.iloc[::N_days, rolling_max.columns == 'month']

                # data_res_proj_cont_sort.iloc[::N_days, 'month'] =
                # df = rolling_mean.iloc[::N_days, :]
                df_all = pd.concat([df, df_m], axis=1)
                col = list(df_all.columns.values)
                m = [col[-1]]
                n_col = col[:-4]+m+col[-4:-1]
                df_all = df_all.reindex(columns=n_col)
                df = df_all[df_all['proj_id'].notna()]
                df_ = pd.concat([df_, df])

    df_.to_excel(path_to_save_dataframe + f'DATA_HOURS_avr_{N_days}_days_corr.xlsx')





    # rolling = data_.rolling(window=N_days)
    # rolling_mean = rolling.mean()
    # df = rolling_mean.iloc[::N_days, :]






