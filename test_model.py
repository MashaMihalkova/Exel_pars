import pandas as pd
import torch
from matplotlib import pyplot as plt
from MODEL.config import ModelType


# предикт по всем данным
def show_results(y_true, y_pred, sum_plans, avr_pred, date_pd, name=None, future=0):
    if not future:
        plt.plot(date_pd, y_true, label='Целевое значение')
    plt.plot(date_pd, y_pred, label='Прогноз ИИ модели', color='orange')
    plt.plot(date_pd, sum_plans, label='Плановое значение Primavera', color='red')

    # plt.plot(avr_pred, label='усредненное значение', color='g')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend(loc='best')
    # plt.xlabel('месяц')
    plt.ylabel('машиночасы')
    if not name: name = 'Cравнение применения ИИ моделей для прогнозирования техники'
    plt.title(name);


def get_predict(PD_DATA, feature_contractor_dict_ids, stages_dict_ids, mech_res_ids, model, model_type, contractor_id,
                contr_id_real, project_id, resource_id, print_result=False, plot_result=True, tracking: int = 0,
                future: int = 0):
    # обычный а
    if future:
        a = list(map(int, range(2, 375)))
    else:
        a = list(map(str, range(0, 373)))
    # Для кс3 с 2023 года

    b = ['proj_id', 'contr_id']
    b.extend(a)
    if tracking:
        b.extend(['day', 'month', 'year', 'res_id', 'target'])
    else:
        if model_type is ModelType.Linear_3MONTH:
            b.extend(['month', 'year', 'res_id', 'm_1', 'm_2', 'm_3', 'target'])
        else:
            b.extend(['month', 'year', 'res_id', 'target'])

    # выгрузка с учетом контрактора
    # df_for_predict = PD_DATA.loc[
    #     (PD_DATA.contr_id == contractor_id) & (PD_DATA.proj_id == project_id) & (PD_DATA.res_id == resource_id), b]

    df_for_predict = PD_DATA.loc[
        (PD_DATA.proj_id == project_id) & (PD_DATA.res_id == resource_id), b]

    # дата по для отрисовки по оси x
    df_for_predict = df_for_predict.sort_values(['year', 'month'])
    date_pd = df_for_predict.copy()
    if df_for_predict.proj_id.unique() == 17:
        if future:
            date_pd['year'] = date_pd['year'].map({0: 2022, 1: 2023})
    else:  # if df_for_predict.proj_id.unique() == 23:
        date_pd['year'] = date_pd['year'].map({0: 2021, 1: 2022})
    date_pd['month'] = date_pd['month'] + 1
    date_pd['date'] = pd.to_datetime(date_pd[['year', 'month']].assign(DAY=1))
    # датафрейм для сохранения в xlsx
    predict_pd_xlsx = pd.DataFrame(columns=('date', 'project', 'predict', 'resource', 'resource_id'))

    features_for_predict = torch.tensor(df_for_predict.iloc[:, :-1].values).to(torch.float)
    target = df_for_predict.iloc[:, -1].values
    if tracking:
        plans = df_for_predict.iloc[:, 2:-1]
    else:
        if model_type is ModelType.Linear_3MONTH:
            plans = df_for_predict.iloc[:, 2:-4]
        else:
            plans = df_for_predict.iloc[:, 2:-1]

    sum_plans = plans.sum(axis=1)

    sum_plans = sum_plans.values
    # sum_plans = sum_plans//3600
    predict = []

    with torch.no_grad():
        for month in range(features_for_predict.shape[0]):
            a = model(features_for_predict[month])
            predict.append(a.tolist())

    if print_result:
        if target.shape[0] != 0:
            plan_targ_predict = {'resource': [], 'month': [], 'predict': [], 'target': [], 'plan': []}
            plan_targ_predict['resource'].append(mech_res_ids[resource_id])
            for i in range(len(predict)):
                # print(f'month = {i+1}, predict = {predict[i] :.2f}, target = {target[i] :.2f}')
                plan_targ_predict['month'].append(i)
                plan_targ_predict['predict'].append(predict[i])
                plan_targ_predict['target'].append(target[i])
                plan_targ_predict['plan'].append(sum_plans[i])
            print(plan_targ_predict)
            return plan_targ_predict
        else:
            return 0

    if plot_result:
        if target.shape[0] != 0:
            print(f'Resource_id = {resource_id}')
            Contractor = list(feature_contractor_dict_ids.keys())[
                list(feature_contractor_dict_ids.values()).index(contr_id_real)]
            project = list(stages_dict_ids.keys())[list(stages_dict_ids.values()).index(project_id)]
            resource = list(mech_res_ids.keys())[list(mech_res_ids.values()).index(resource_id)]
            pred_serias = pd.Series(predict)
            rolling = pred_serias.rolling(window=5)
            rolling_mean = rolling.mean()
            # print(rolling_mean.head(10))
            # date_pd['date']

            predict_pd_xlsx['date'] = date_pd['date']
            predict_pd_xlsx['project'] = project
            predict_pd_xlsx['predict'] = predict
            predict_pd_xlsx['resource'] = resource
            predict_pd_xlsx['resource_id'] = resource_id

            show_results(target, predict, sum_plans, rolling_mean, date_pd['date'],
                         name=f'Contractor = {Contractor},'
                              f' project = {project} \n resource = {resource}', future=future)
        return predict_pd_xlsx
