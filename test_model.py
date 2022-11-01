import torch
from matplotlib import pyplot as plt


def show_results(y_true, y_pred, name=None):
    plt.plot(y_true, label='Целевое значение')
    plt.plot(y_pred, label='Прогноз ИИ модели')

    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('месяцы')
    plt.ylabel('машиночасы')
    if not name: name = 'Cравнение применения ИИ моделей для прогнозирования техники'
    plt.title(name);


def get_predict(PD_DATA, feature_contractor_dict_ids, stages_dict_ids, mech_res_ids,  model, contractor_id, project_id,
                resource_id, print_result=False, plot_result=True):
    a = list(map(str, range(0, 373)))
    b = ['proj_id', 'contr_id']
    b.extend(a)
    b.extend(['month', 'year', 'res_id', 'target'])
    df_for_predict = PD_DATA.loc[(PD_DATA.contr_id == contractor_id) & (PD_DATA.proj_id == project_id) & (
            PD_DATA.res_id == resource_id), b]
    features_for_predict = torch.tensor(df_for_predict.iloc[:, :-1].values).to(torch.float)
    target = df_for_predict.iloc[:, -1].values

    predict = []

    with torch.no_grad():
        for month in range(features_for_predict.shape[0]):
            a = model(features_for_predict[month])
            predict.append(a.tolist())

    if print_result:
        for i in range(len(predict)):
            print(f'month = {i + 1}, predict = {predict[i] :.2f}, target = {target[i] :.2f}')

    if plot_result:
        show_results(target, predict,
                     name=f'Contractor = {feature_contractor_dict_ids[contractor_id]},'
                          f' project = {stages_dict_ids[project_id]} ,\n resource_id = {mech_res_ids[resource_id]} ')
