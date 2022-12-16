import torch.cuda

from create_dataset.averaging_over_days import averaging_over_days
from targets_parsing.pars_targets import *
from targets_parsing.convert_to_target_npy import *
from targets_parsing.tracking_target_pars import *
from sql_connection.extract_target_tracking_from_bd import *
from sql_connection.queary_to_bd import *
# from sql_connection.databese import *  # подключение через SQLAlchemy, не выполняте запрос...
from sql_connection.extract_data_from_bd import sql_quary
from create_dataset.create_dataset_ import *
from glob import glob
from train_model import train_model, train
from wand_config.config import *
# from datetime import datetime
from create_dop_materials.status_tech import *
from create_dop_materials.create_stages import *
from create_dop_materials.create_contractors import *
from create_dataset.converter_from_database_loader_to_npy import *
from create_dataset.add_statistic_3month import add_statistic_100percent, add_statistic_previous
import optparse
from MODEL.config import Parameters, ModelType, CriteriaType
from Log.print_lib import *
import os
import pandas as pd
import numpy as np
from test_model import *

"""
    ПОДГОТОВКА ДАННЫХ
        0. чтение из бд по ид_проекта
        1. создать stage.npy contractor.npy и тд
        2. преобразуем полученные данные в нужный вид (proj_id, contr_id, PO, month, year, res_id) len = 378
            сохранение в npy (но мб можно и упустить это
    ПОДГОТОВКА ТАРГЕТА
        3. Создаем/Читаем таргет
    СОЗДАНИЕ БД
        4. запускаем create_dataset
"""


def create_dataloaders_train_test(pd_data, model_type):  # noqa
    test_PD_DATA = pd_data.loc[(pd_data['month'] == 7) | (pd_data['month'] == 8)]
    test_PD_DATA = test_PD_DATA.loc[test_PD_DATA['year'] == 1]
    train_PD_DATA = pd_data[~pd_data.index.isin(test_PD_DATA.index)]
    train_PD_DATA = train_PD_DATA.sort_values(['year', 'month'])
    test_PD_DATA = test_PD_DATA.sort_values(by=['year'])

    print_i(f'ModelType = {model_type}')
    if model_type is ModelType.LSTM:
        mech_res_dict = np.load(DOP_DATA_PATH + 'mech_res_dict.npy', allow_pickle=True).item()  # noqa
        train_dataset = PROJDataset_sequenses(train_PD_DATA, mech_res_dict)  # noqa
        test_dataset = PROJDataset_sequenses(test_PD_DATA, mech_res_dict)  # noqa
    elif model_type is ModelType.Linear or model_type is ModelType.Linear_3MONTH:
        train_dataset = PROJDataset(train_PD_DATA)  # noqa
        test_dataset = PROJDataset(test_PD_DATA)  # noqa
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)  # noqa
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # noqa
    return train_loader, test_loader


def download_projects(_id: int, FO: str, stage_proj_name: str) -> int:
    _df = pd.read_sql_query(sql_quary(proj_id=_id, FO=FO), cnxn)
    if not _df.empty:
        _flag = 1
        # _df.to_excel(PATH_EXCEL_PROJECTS + list(_df['project_name'].unique())[0] + '.xlsx')
        _df.to_excel(PATH_EXCEL_PROJECTS + str(stage_proj_name) + '.xlsx')
        # print_i(
        #     f"SUCCESS SAVE UNLOADING TO {PATH_EXCEL_PROJECTS + list(_df['project_name'].unique())[0] + '.xlsx'}")
        print_i(
            f"SUCCESS SAVE UNLOADING TO {PATH_EXCEL_PROJECTS + str(stage_proj_name) + '.xlsx'}")
    else:
        print_e(f"DATA WITH PROJECT_ID = {_id} IS EMPTY!")
        _flag = 0
    return _flag


if __name__ == '__main__':
    # TODO: сделать предик, брав данные с бд!
    parser = optparse.OptionParser()

    parser.add_option('-l', '--DOWNLOAD_ALL_PROJECTS_FROM_DB', type=int,
                      help="DOWNLOAD_ALL_PROJECTS_FROM_DB instead of unloading a single project by id, unload all",
                      default=0)

    parser.add_option('-t', '--PATH_TO_TARGETS_EXCEL', type=str,
                      help="PATH_TO_TARGETS_EXCEL", default='data/targets_excel/')

    parser.add_option('-s', '--PATH_TO_SAVE_TARGETS', type=str,
                      help="PATH_TO_SAVE_TARGETS or PATH TO LOAD TARGET IF IT`S EXISTS", default='data/')

    parser.add_option('-j', '--PATH_TO_PROJECTS', type=str, help="PATH_TO_PROJECTS", default='data/')

    parser.add_option('-p', '--PATH_EXCEL_PROJECTS', type=str, help="PATH_EXCEL_PROJECTS", default='data/predict_data/features/')

    parser.add_option('-n', '--PATH_NPY_PROJECTS', type=str,
                      help="PATH_NPY_PROJECTS", default='data/predict_data/features/')  # data/prepred_train_data/

    parser.add_option('-w', '--SAVE_WEIGHT', type=str, help="SAVE_WEIGHT", default='data/WEIGHTS/')

    parser.add_option('-d', '--DOP_DATA_PATH', type=str, help="DOP_DATA_PATH", default='data/dop_materials/')

    parser.add_option('-c', '--CONNECT', type=int, help="CONNECT to db", default=0)

    # parser.add_option('-i', '--PROJ_ID', type=int, help="PROJ_ID", default=32159)
    parser.add_option('-i', '--PROJ_ID', type=str, help="PROJ_ID", default='28393')

    parser.add_option('-r', '--RELOAD_DOPS', type=int,
                      help="NEED TO RELOAD_DOPS IF IT`S CHANGED OR IT DOESNT EXIST", default=0)

    parser.add_option('-v', '--CONVERT', type=int, help="CONVERT PROJECT TO NUMPY", default=0)

    parser.add_option('-g', '--TARGET', type=int, help="NEED TO PARSING TARGETS", default=0)

    parser.add_option('-e', '--TARGET_TRACKING', type=int, help="NEED TO PARSING TARGETS", default=0)

    parser.add_option('-m', '--TARGET_CONNECT', type=int, help="NEED TO DOWNLOAD TARGETS TRACKING FROM DB", default=0)

    parser.add_option('-a', '--CREATE_DATASET', type=int, help="CREATE_DATASET", default=0)

    parser.add_option('-z', '--ADD_STATISTIC', type=int, help="ADD_STATISTIC to dataset 3 month", default=0)

    options, args = parser.parse_args()
    DOWNLOAD_ALL_PROJECTS_FROM_DB = getattr(options, 'DOWNLOAD_ALL_PROJECTS_FROM_DB')

    PATH_TO_TARGETS_EXCEL = getattr(options, 'PATH_TO_TARGETS_EXCEL')
    PATH_TO_SAVE_TARGETS = getattr(options, 'PATH_TO_SAVE_TARGETS')
    PATH_TO_PROJECTS = getattr(options, 'PATH_TO_PROJECTS')
    PATH_EXCEL_PROJECTS = getattr(options, 'PATH_EXCEL_PROJECTS')
    PATH_NPY_PROJECTS = getattr(options, 'PATH_NPY_PROJECTS')
    SAVE_WEIGHT = getattr(options, 'SAVE_WEIGHT')
    DOP_DATA_PATH = getattr(options, 'DOP_DATA_PATH')

    CONNECT = getattr(options, 'CONNECT')
    # PROJ_ID = getattr(options, 'PROJ_ID')
    PROJ_ID = [int(item) for item in getattr(options, 'PROJ_ID').split(',')]
    RELOAD_DOPS = getattr(options, 'RELOAD_DOPS')
    CONVERT = getattr(options, 'CONVERT')
    TARGET = getattr(options, 'TARGET')
    TARGET_TRACKING = getattr(options, 'TARGET_TRACKING')
    TARGET_CONNECT = getattr(options, 'TARGET_CONNECT')
    CREATE_DATASET = getattr(options, 'CREATE_DATASET')
    ADD_STATISTIC = getattr(options, 'ADD_STATISTIC')

    TRAIN: int = 0
    TEST: int = 1
    PREDICT_FUTURE = 1
    if PREDICT_FUTURE:
        FO = 'remain_qty'
    else:
        FO = 'act_reg_qty'

    WandB_flag = 0  # запуск wandb

    avarage_hours = 0  # усреднить показания с двигателя omni по часам ( т е либо 10 либо 24 часа отрабатывала техника),
                       # чтобы избавиться от шума

    create_averaging_over_N_days = 0
    averaging_over_N_days = 0  # усреднение показаний с двигателя omni по часам по N дней
                               # (т.е. предсказывать среднее значение на N дней)
    N_days_avr = 5

    # дополнительный файл со статусами из КА
    STATUS_TECH = 0  # переименование ид IDINTROBJ в объект.
    PATH_TO_STATUS = 'data/status_tech/tech_status.xlsx'
    PATH_TO_ID_INTROBJ = 'data/status_tech/IDINTROBJ_id.xlsx'
    PATH_TO_SAVE_STATUS = 'data/status_tech/RENAME_IDINTROBJ.xlsx'

    previous = 1  # add statistic with previous = 1, add statistic with percent = 0
    BATCH_SIZE: int = 8
    LR: float = 0.001
    EPOCH: int = 500
    NW: int = 0
    L2: float = 0.0001
    flag = 0
    error_flag = 0
    config = {
        "learning_rate": LR,
        "epochs": EPOCH,
        "batch_size": BATCH_SIZE,
        "num_workers": NW,
        "weight_decay(l2)": L2,

    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_column_fact = 13  # 17 - если хотим по omni, 13 - если по времени
    dosaz = 0

    # model_type = ModelType.Linear_3MONTH
    model_type = ModelType.Linear
    criteria_type = CriteriaType.HuberLoss

    for dir_ in [PATH_TO_SAVE_TARGETS, PATH_TO_PROJECTS, PATH_EXCEL_PROJECTS, PATH_NPY_PROJECTS,
                 SAVE_WEIGHT, DOP_DATA_PATH, PATH_TO_TARGETS_EXCEL]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    if not os.path.exists(DOP_DATA_PATH + 'stages.xlsx'):
        create_stages(DOP_DATA_PATH)
        print_i(f'в папке {DOP_DATA_PATH} создался файл stages.xlsx')

    if not os.path.exists((DOP_DATA_PATH + 'all_contractors.xlsx')):
        create_contractors(DOP_DATA_PATH)
        print_i(f'в папке {DOP_DATA_PATH} создался файл all_contractors.xlsx')

    # region LOAD data from db
    # if NEED_LOAD_FROM_DB:
    # 0 CONNECTION to DB
    # region Connect to db
    if CONNECT:
        stages_data = pd.DataFrame(pd.read_excel(DOP_DATA_PATH + 'stages.xlsx'))  # noqa
        if DOWNLOAD_ALL_PROJECTS_FROM_DB:
            print_i(f"TRY TO CONNECT TO DB AND DOWNLOAD ALL PROJECTS")
            stages_id = list(stages_data.id_project)
            stages_pr_name = list(stages_data.project_name)
            for ind, _id in enumerate(stages_id):
                print_i(f"PROJECT_ID = {_id}")
                flag = download_projects(_id, FO, stages_pr_name[ind])  # noqa
                flag = 1  # noqa
        else:
            if isinstance(PROJ_ID, list):
                for _id in PROJ_ID:
                    stages_pr_name = stages_data.loc[stages_data['id_project'] == _id, 'project_name']
                    print_i(f"PROJECT_ID = {_id}")
                    flag = download_projects(_id, FO, stages_pr_name.values[0])
            else:
                print_i(f'TRY TO CONNECT TO DB, PROJ_ID = {PROJ_ID[0]}')
                flag = download_projects(PROJ_ID[0], FO, 'none')
    # endregion

    # region RELOAD CONTRACTOR.npy MECH.npy STAGE.npy
    # 1 CREATE CONTRACTOR.npy MECH.npy STAGE.npy
    if RELOAD_DOPS:
        print_i(f'TRY TO RELOAD: mech_res_dict, stages_dict')
        assert len([name for name in os.listdir(DOP_DATA_PATH) if os.path.isfile(os.path.join(DOP_DATA_PATH, name))]
                   ) > 0, print_e("Закинь Norms.xlsx в папку Needed_materials")
        df_norm = pd.read_excel('data/Needed_materials/Norms.xlsx')  # noqa
        mech_res_list = pd.unique(df_norm.loc[pd.notna(df_norm.iloc[:, 4]), 'Работа\специальность\техника'])  # noqa
        mech_res_dict = dict(zip(mech_res_list, range(len(mech_res_list))))
        np.save(DOP_DATA_PATH + 'mech_res_dict.npy', mech_res_dict, allow_pickle=True)
        df_stages = pd.read_excel(DOP_DATA_PATH + 'stages.xlsx')  # noqa
        stages_dict = df_stages.set_index('project_name').id.to_dict()
        np.save(DOP_DATA_PATH + 'stages_dict.npy', stages_dict, allow_pickle=True)
        print_i('SUCCESS RELOADING')
    # endregion

    # region convert projects to npy
    # 2 CONVERT to proj_data(len = 378)
    if CONVERT:
        # if PREDICT_FUTURE:
        #     print_i(f'TRY TO CONVERT XLSX PROJECTS TO NPY FOR PREDICT')
        #     assert len([name for name in os.listdir(DOP_DATA_PATH) if
        #                 os.path.isfile(os.path.join(DOP_DATA_PATH, name))]) > 1, print_e(
        #         "Закинь данные в папку для доп материала")
        #     feature_contractor_dict = np.load(DOP_DATA_PATH + 'all_contractors.npy', allow_pickle=True).item()
        #     mech_res_dict = np.load(DOP_DATA_PATH + 'mech_res_dict.npy', allow_pickle=True).item()
        #     stages_dict = np.load(DOP_DATA_PATH + 'stages_dict.npy', allow_pickle=True).item()
        #     for i, path in enumerate(glob(PATH_EXCEL_PROJECTS + '*.xlsx')):
        #         print(i, path)
        #         df = pd.read_excel(path)  # noqa
        #         df['dt'] = df.dt.apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
        #         df['year'] = df.dt.dt.year
        #         df['month'] = df.dt.dt.month
        #         df['day'] = df.dt.dt.day
        #         df['res_id'] = df.resource_name.map(mech_res_dict)
        #         df['contr_id'] = df.contractor_name.map(feature_contractor_dict)
        #         prepare_features(df, path, save_path=PATH_NPY_PROJECTS, stages_dict=stages_dict,
        #                          tracking=TARGET_TRACKING)
        #     print_i('SUCCESS CONVERT PROJECTS TO NPY')
        # else:
        print_i(f'TRY TO CONVERT XLSX PROJECTS TO NPY')
        assert len([name for name in os.listdir(DOP_DATA_PATH) if
                    os.path.isfile(os.path.join(DOP_DATA_PATH, name))]) > 1, print_e(
            "Закинь данные в папку для доп материала")
        feature_contractor_dict = np.load(DOP_DATA_PATH + 'all_contractors.npy', allow_pickle=True).item()
        mech_res_dict = np.load(DOP_DATA_PATH + 'mech_res_dict.npy', allow_pickle=True).item()
        stages_dict = np.load(DOP_DATA_PATH + 'stages_dict.npy', allow_pickle=True).item()
        for i, path in enumerate(glob(PATH_EXCEL_PROJECTS + '*.xlsx')):
            print(i, path)
            df = pd.read_excel(path)  # noqa
            df['dt'] = df.dt.apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
            df['year'] = df.dt.dt.year
            df['month'] = df.dt.dt.month
            df['day'] = df.dt.dt.day
            df['res_id'] = df.resource_name.map(mech_res_dict)
            df['contr_id'] = df.contractor_name.map(feature_contractor_dict)
            prepare_features(df, path, save_path=PATH_NPY_PROJECTS, stages_dict=stages_dict, tracking=TARGET_TRACKING,
                             FO=FO)
        print_i('SUCCESS CONVERT PROJECTS TO NPY')
    # endregion
    # endregion

    # region NEED CREATE TARGET
    if TARGET:
        if not TARGET_TRACKING:
            print_i('TRY TO CREATE TARGET')
            if len([name for name in os.listdir(PATH_TO_TARGETS_EXCEL) if
                    os.path.isfile(os.path.join(PATH_TO_TARGETS_EXCEL, name))]) == 0:
                print_e(f"THERE IS NOT ANY FILE IN {PATH_TO_TARGETS_EXCEL}")
                error_flag = 1
            else:
                a = pd.DataFrame()
                year = 0
                i = 0
                for i, path_ in enumerate(glob(PATH_TO_TARGETS_EXCEL + '*.xls')):
                    print_d(i, path_)
                    dataframe = pd.read_excel(path_)  # noqa
                    if os.path.splitext(path_)[-1] == '.xls':
                        if dosaz:
                            dat = parse_data_xls(dataframe, fact=7)
                        else:
                            dat = parse_data_xls(dataframe, fact=num_column_fact)
                    else:
                        dat = parse_data(dataframe, fact=num_column_fact)
                    # dat = parse_data_xls(dataframe)

                    year = int(dataframe.iloc[0, 2][14:18])
                    month = int(dataframe.iloc[0, 2][11:13])

                    if i == 0:
                        a = pd.DataFrame(dat)
                    a = pd.concat([a, pd.DataFrame(dat)], axis=0, join='outer', ignore_index=True)

                for ii, path_ in enumerate(glob(PATH_TO_TARGETS_EXCEL + '*.xlsx')):
                    print_d(ii, path_)
                    dataframe = pd.read_excel(path_)  # noqa
                    if os.path.splitext(path_)[-1] == '.xlsx':
                        dat = parse_data(dataframe, fact=num_column_fact)
                    year = int(dataframe.iloc[0, 2][14:18])
                    month = int(dataframe.iloc[0, 2][11:13])

                    if i == 0:
                        a = pd.DataFrame(dat)
                    a = pd.concat([a, pd.DataFrame(dat)], axis=0, join='outer', ignore_index=True)

                a.to_excel(f'{PATH_TO_SAVE_TARGETS}whole_{year}.xlsx')
                convert_target_to_npy(f'{PATH_TO_SAVE_TARGETS}whole_{year}.xlsx', DOP_DATA_PATH, PATH_TO_SAVE_TARGETS)
                print_i(f"SUCCESS CREATE TARGET.NPY IN {PATH_TO_SAVE_TARGETS}")
    # endregion

    # region TARGET_TRACKING
    if TARGET:
        if TARGET_TRACKING:
            print_i('TRY TO CREATE TARGET_TRACKING')
            if TARGET_CONNECT:
                print_i("TRY TO CONNECT TO THE DATABASE TO UPLOAD TARGETS TRACKING")
                df = pd.read_sql_query(sql_quary_target(), cnxn_target)
                df.to_excel(PATH_TO_SAVE_TARGETS + 'omnicom_data.xlsx')
            else:
                print_i("OPEN TARGETS TRACKING FILE omnicom_data.xlsx ")
                df = pd.read_excel(PATH_TO_SAVE_TARGETS + 'omni_data.xlsx')
                # df = pd.read_excel(PATH_TO_SAVE_TARGETS + 'omnicom_data.xlsx')

            tracking_target_pars(df, path_to_dop_materials='data/Needed_materials/', path_to_save=PATH_TO_SAVE_TARGETS,
                                 avarage_hours=avarage_hours)
            if avarage_hours:
                convert_target_to_npy(f'{PATH_TO_SAVE_TARGETS}whole_target_hours_omni_10-24.xlsx', DOP_DATA_PATH,
                                      PATH_TO_SAVE_TARGETS, tracking=1, avarage_hours=avarage_hours)
            else:
                convert_target_to_npy(f'{PATH_TO_SAVE_TARGETS}whole_target_hours_omni.xlsx', DOP_DATA_PATH,
                                      PATH_TO_SAVE_TARGETS, tracking=1)
            print_i(f"SUCCESS CREATE whole_target_hours_omni.NPY IN {PATH_TO_SAVE_TARGETS}")
    # endregion

    # region Create DATASET
    if CREATE_DATASET:
        # TODO: надо отредактировать контракторов и месяца (месяца сделать 12,
        #  контракторов пронумеровать с 0 (сейчас есть один контрактор его ид=14, а дб 0)
        print_i('TRY TO CREATE DATASET')
        if error_flag:
            print_e(f'THERE IS NO target_array.npy IN {PATH_TO_SAVE_TARGETS}')
        else:
            if TARGET_TRACKING:
                if avarage_hours:
                    targets = np.load(PATH_TO_SAVE_TARGETS + 'whole_target_hours_omni_HOURS.npy')
                else:
                    targets = np.load(PATH_TO_SAVE_TARGETS + 'whole_target_hours_omni.npy')
            else:
                targets = np.load(PATH_TO_SAVE_TARGETS + 'target_array.npy')
            print_i(f'TAKE TARGET FROM {PATH_TO_SAVE_TARGETS}')
            PD_tar = pd.DataFrame(targets)  # 0/1-contr/proj, 2-month, 3- year, 4 - res, 5 - val
            if PD_tar.empty:
                error_flag = 1
        if TARGET_TRACKING:
            Projects = list(glob(PATH_NPY_PROJECTS + '/tracking/*.npy'))[0:]
        else:
            Projects = list(glob(PATH_NPY_PROJECTS + '/*.npy'))[0:]
        if len(Projects) > 0 and not error_flag:
            dict_data = create_dataset(Projects, PD_tar, tracking=TARGET_TRACKING)  # noqa
            PD_DATA = pd.DataFrame(dict_data)
            # TODO: удалить первую колонку если она не proj_id
            # PD_DATA = PD_DATA.drop()
            #
            if TARGET_TRACKING:
                PD_DATA.to_excel(PATH_TO_PROJECTS + 'DATA_HOURS_correct.xlsx')
            else:
                PD_DATA.to_excel(PATH_TO_PROJECTS + 'DATA.xlsx')
            if ADD_STATISTIC:
                if previous:
                    print_i('ADD 3 MONTH STATISTIC with previous')
                    Stat_PD_DATA = add_statistic_previous(PD_DATA)
                    Stat_PD_DATA.to_excel(PATH_TO_PROJECTS + 'Stat_PD_DATA_previous.xlsx')
                else:
                    print_i('ADD 3 MONTH STATISTIC with percent')
                    Stat_PD_DATA = add_statistic_100percent(PD_DATA)
                    Stat_PD_DATA.to_excel(PATH_TO_PROJECTS + 'Stat_PD_DATA_percent.xlsx')
        else:
            if error_flag:
                print_e(f"ERROR WITH TARGETS WHEN TRY TO CREATE DATASET, BECAUSE error_flag = {error_flag} ")
            else:
                print_e(f"ERROR WHEN TRY TO CREATE DATASET, BECAUSE THERE ARE NO FILES IN PATH_NPY_PROJECTS")
                error_flag = 1
    else:
        if avarage_hours:
            if os.path.exists(f'{PATH_TO_PROJECTS}DATA_HOURS_correct.xlsx'):
                print_i(f'TAKE DATASET FROM {PATH_TO_PROJECTS}DATA_HOURS_correct.xlsx')
            else:
                print_e(f"THERE ARE NO ANY FILES IN {PATH_TO_PROJECTS}DATA_HOURS_correct.xlsx ")
                error_flag = 1
        else:
            if os.path.exists(f'{PATH_TO_PROJECTS}DATA.xlsx'):
                print_i(f'TAKE DATASET FROM {PATH_TO_PROJECTS}DATA.xlsx')
            else:
                print_e(f"THERE ARE NO ANY FILES IN {PATH_TO_PROJECTS}DATA.xlsx ")
                error_flag = 1
    # endregion

    # region averaging_over_N_days
    if create_averaging_over_N_days:
        print_i(f'TRY to average dataset DATA_HOURS_correct.xlsx over {N_days_avr} days')
        averaging_over_days(pd.read_excel(PATH_TO_PROJECTS + 'DATA_HOURS_correct.xlsx'), N_days_avr, PATH_TO_PROJECTS)

    # endregion

    # region TRAIN model
    if TRAIN:
        # load data
        print(torch.cuda.is_available())
        if not error_flag:
            if ADD_STATISTIC:
                Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS + 'Stat_PD_DATA.xlsx')  # noqa
            else:
                if avarage_hours:
                    # Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS + 'DATA_HOURS.xlsx')  # noqa
                    Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS + 'DATA_HOURS_correct.xlsx')  # noqa
                else:
                    Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS + 'DATA_.xlsx')  # noqa
            if 'Unnamed: 0' in Stat_PD_DATA.columns:
                Stat_PD_DATA = Stat_PD_DATA.drop('Unnamed: 0', axis=1)
            # TODO: сделать нормальное преобразование OneHotEncoder get_dummies
            # convert contractor, month and year
            uniq_contr = Stat_PD_DATA['contr_id'].unique()
            for i, contr in enumerate(uniq_contr):
                Stat_ = Stat_PD_DATA.loc[Stat_PD_DATA['contr_id'] == contr]
                if not Stat_.empty:
                    Stat_['contr_id'] = i
            Stat_PD_DATA = Stat_

            uniq_month = Stat_PD_DATA['month'].unique()
            Stat_PD_DATA_m = pd.DataFrame()
            for i, month in enumerate(uniq_month):
                Stat_m = Stat_PD_DATA.loc[Stat_PD_DATA['month'] == month]
                Stat_m['month'] = month - 1
                if month > 12:
                    Stat_m['month'] = month - 12
                    # Stat_PD_DATA.loc[Stat_PD_DATA['month'] == month] = month - 12
                Stat_PD_DATA_m = Stat_PD_DATA_m.append(Stat_m)
            Stat_PD_DATA = Stat_PD_DATA_m

            uniq_year = Stat_PD_DATA['year'].unique()
            Stat_PD_DATA_y = pd.DataFrame()
            for i, year in enumerate(uniq_year):
                Stat_y = Stat_PD_DATA.loc[Stat_PD_DATA['year'] == year]
                if year == 2022:
                    Stat_y['year'] = 1
                else:
                    Stat_y['year'] = 0
                    # Stat_PD_DATA.loc[Stat_PD_DATA['month'] == month] = month - 12
                Stat_PD_DATA_y = Stat_PD_DATA_y.append(Stat_y)
            Stat_PD_DATA = Stat_PD_DATA_y

            # create dataset
            train_loader, test_loader = create_dataloaders_train_test(Stat_PD_DATA, model_type)
            # create model
            model_param = Parameters(config, model_type, criteria_type)
            # load pretrain model
            model_param.net.load_state_dict(torch.load(
                'data/WEIGHTS/avr_hour/add_project_to_model/best_pretrained_weights.pt'))
            # run wandb
            if WandB_flag:
                WandB(config, model_type, criteria_type)
            # train model
            model = train(model_param.net, train_loader, test_loader, model_param.criteria, 0, model_param.optimizer, 0,
                          model_param.epochs, SAVE_WEIGHT, device, 'name')
            torch.save(model.state_dict(), f"{SAVE_WEIGHT}model.pt")
            print(f"Done. Model saved in folder [{SAVE_WEIGHT}]")
    # endregion

    # region TEST MODEL
    if TEST:
        # load data
        # print(torch.cuda.is_available())
        if not error_flag:
            if PREDICT_FUTURE:
                Stat_PD_DATA = pd.DataFrame()
                stages_dict = np.load(DOP_DATA_PATH + 'stages_dict.npy', allow_pickle=True).item()
                for i, path_ in enumerate(glob(PATH_NPY_PROJECTS + '*.npy')):
                    name_proj = path_.split('\\')[-1][:-4]
                    PD_DATA = pd.DataFrame(np.load(path_, allow_pickle=True))
                    PD_DATA = PD_DATA.fillna(0)
                    PD_DATA['proj_id'] = stages_dict.get(name_proj)

                    cols = PD_DATA.columns.tolist()
                    col_n = [cols[-1]] + cols[1:-1] + [cols[0]]
                    PD_DATA = PD_DATA[col_n]

                    PD_DATA = PD_DATA.rename({1: 'contr_id', 375: 'month', 376: 'year', 377: 'res_id', 0: 'target'}, axis=1)
                    Stat_PD_DATA = pd.concat([Stat_PD_DATA, PD_DATA], axis=0)

            else:
                if ADD_STATISTIC:
                    Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS + 'Stat_PD_DATA.xlsx')  # noqa
                else:
                    if avarage_hours:
                        if averaging_over_N_days:
                            Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS + 'DATA_HOURS_avr_5_days_corr.xlsx')  # noqa
                        else:
                            Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS + 'DATA_HOURS_correct.xlsx')  # noqa
                    else:
                        Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS + 'DATA.xlsx')  # noqa

            if 'Unnamed: 0' in Stat_PD_DATA.columns:
                Stat_PD_DATA = Stat_PD_DATA.drop('Unnamed: 0', axis=1)
            # convert

            df_for_real_contractor = Stat_PD_DATA.copy()
            uniq_contr = Stat_PD_DATA['contr_id'].unique()
            Stat_PD_DATA_c = pd.DataFrame()
            for i, contr in enumerate(uniq_contr):
                Stat_ = Stat_PD_DATA.loc[Stat_PD_DATA['contr_id'] == contr]
                if not Stat_.empty:
                    Stat_['contr_id'] = i+1
                Stat_PD_DATA_c = Stat_PD_DATA_c.append(Stat_)
            Stat_PD_DATA = Stat_PD_DATA_c
            contractor_id = list(Stat_PD_DATA['contr_id'].unique())

            uniq_month = Stat_PD_DATA['month'].unique()
            Stat_PD_DATA_m = pd.DataFrame()
            for i, month in enumerate(uniq_month):
                Stat_m = Stat_PD_DATA.loc[Stat_PD_DATA['month'] == month]
                Stat_m['month'] = month - 1
                if month > 12:
                    Stat_m['month'] = month - 12
                    # Stat_PD_DATA.loc[Stat_PD_DATA['month'] == month] = month - 12
                Stat_PD_DATA_m = Stat_PD_DATA_m.append(Stat_m)
            Stat_PD_DATA = Stat_PD_DATA_m

            # uniq_year = Stat_PD_DATA['year'].unique()
            Stat_PD_DATA_y = pd.DataFrame()
            for proj_id in Stat_PD_DATA['proj_id'].unique():
                proj_Stat_PD_DATA = Stat_PD_DATA.loc[Stat_PD_DATA['proj_id'] == proj_id, :]
                if proj_id == 17:  # убрать 2021 год для кс3
                    proj_Stat_PD_DATA = proj_Stat_PD_DATA[proj_Stat_PD_DATA['year'] != 2021]
                    proj_Stat_PD_DATA['year'] = proj_Stat_PD_DATA['year'].map({2022: 0, 2023: 1})
                else:
                    proj_Stat_PD_DATA['year'] = proj_Stat_PD_DATA['year'].map({2021: 0, 2022: 1})
                Stat_PD_DATA_y = Stat_PD_DATA_y.append(proj_Stat_PD_DATA)
            Stat_PD_DATA = Stat_PD_DATA_y

            # create dataset
            train_loader, test_loader = create_dataloaders_train_test(Stat_PD_DATA, model_type)
            # load dop materials
            feature_contractor_dict = np.load(DOP_DATA_PATH + 'all_contractors.npy', allow_pickle=True).item()
            stages_dict = np.load(DOP_DATA_PATH + 'stages.npy', allow_pickle=True).item()
            mech_res_dict = np.load(DOP_DATA_PATH + 'mech_res_dict.npy', allow_pickle=True).item()
            feature_contractor_dict_ids = {v: k for k, v in feature_contractor_dict.items()}
            stages_dict_ids = {v: k for k, v in stages_dict.items()}
            mech_res_ids = {v: k for k, v in mech_res_dict.items()}
            # create model
            model_param = Parameters(config, model_type, criteria_type)
            # load weights
            if avarage_hours:
                # model_param.net.load_state_dict(torch.load(f"{SAVE_WEIGHT}log_model_huber_05_loss0.588.pt"))
                # model_param.net.load_state_dict(
                #     torch.load(f"{SAVE_WEIGHT}/avr_hour/add_project_to_model/log_model_huber_05_loss0.108.pt"))
                model_param.net.load_state_dict(
                    torch.load(f"{SAVE_WEIGHT}avr_hour/add_project_to_model/log_model_huber_05_loss2.544.pt"))
            else:
                # model_param.net.load_state_dict(torch.load(f"{SAVE_WEIGHT}log_model_huber_05_loss12711.709.pt"))
                model_param.net.load_state_dict(torch.load(f"{SAVE_WEIGHT}log_model_huber_05_epoch_500_loss_69_mae_547.pt"))

            # needed machines
            tech = [2, 5, 8, 14, 19, 29, 30, 32, 42, 44, 46, 48, 57, 65, 70, 74, 76, 77, 83, 101, 111, 112, 115, 125,
                    143, 157, 172, 209, 216, 234, 235]
            # tech = [30]
            common_statist = []
            # choose project
            project_id = 23
            contr_for_proj = df_for_real_contractor.loc[df_for_real_contractor['proj_id'] == project_id]

            contr_id_real = list(contr_for_proj['contr_id'].unique())

            dict_statist_ = pd.DataFrame()
            # run test for project by all contractors and machines
            for ind, c in enumerate(contr_id_real):

                for i in tech:
                    print(i)
                    dict_statist = get_predict(Stat_PD_DATA, feature_contractor_dict, stages_dict, mech_res_dict,
                                               model_param.net, model_type, contractor_id=contractor_id[ind],
                                               contr_id_real=c, project_id=project_id,
                                               resource_id=i, print_result=False, plot_result=True, tracking=TARGET_TRACKING,
                                               future=PREDICT_FUTURE)

                    plt.show()
                    # if dict_statist != 0:
                    #     common_statist.append(dict_statist)
                    dict_statist_ = pd.concat([dict_statist, dict_statist_], axis=0)
            dict_statist_.to_excel(f'data/predict_data/features/predict_NN/predict_proj_id_{project_id}.xlsx')
    # endregion

    # region Status tech
    if STATUS_TECH:
        status_tech(PATH_TO_STATUS, PATH_TO_ID_INTROBJ, PATH_TO_SAVE_STATUS)
    # endregion
