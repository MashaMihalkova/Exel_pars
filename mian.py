from targets_parsing.pars_targets import *
from targets_parsing.convert_to_target_npy import *
from sql_connection.queary_to_bd import *
from sql_connection.extract_data_from_bd import sql_quary
from create_dataset.create_dataset_ import *
from glob import glob
from train_model import train_model
# from wand_config.config import *
# from datetime import datetime
from create_dop_materials.create_stages import *
from create_dataset.converter_from_database_loader_to_npy import *
from create_dataset.add_statistic_3month import add_statistic
import optparse
from MODEL.config import Parameters, ModelType, CriteriaType
from Log.print_lib import *
import os
import pandas as pd
import numpy as np

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

if __name__ == '__main__':

    parser = optparse.OptionParser()

    parser.add_option('-l', '--NEED_LOAD_FROM_DB', type=int, help="NEED_LOAD_FROM_DB", default=1)

    parser.add_option('-t', '--PATH_TO_TARGETS_EXCEL', type=str,
                      help="PATH_TO_TARGETS_EXCEL", default='data_/targets_excel/')

    parser.add_option('-s', '--PATH_TO_SAVE_TARGETS', type=str,
                      help="PATH_TO_SAVE_TARGETS or PATH TO LOAD TARGET IF IT`S EXISTS", default='data')

    parser.add_option('-j', '--PATH_TO_PROJECTS', type=str, help="PATH_TO_PROJECTS", default='data/')

    parser.add_option('-p', '--PATH_EXCEL_PROJECTS', type=str, help="PATH_EXCEL_PROJECTS", default='data/features/')

    parser.add_option('-n', '--PATH_NPY_PROJECTS', type=str,
                      help="PATH_NPY_PROJECTS", default='data/prepred_train_data/')

    parser.add_option('-w', '--SAVE_WEIGHT', type=str, help="SAVE_WEIGHT", default='data/WEIGHTS/')

    parser.add_option('-d', '--DOP_DATA_PATH', type=str, help="DOP_DATA_PATH", default='data/dop_materials/')

    parser.add_option('-c', '--CONNECT', type=int, help="CONNECT to db", default=1)

    parser.add_option('-i', '--PROJ_ID', type=int, help="PROJ_ID", default=32159)

    parser.add_option('-r', '--RELOAD_DOPS', type=int,
                      help="NEED TO RELOAD_DOPS IF IT`S CHANGED OR IT DOESNT EXIST", default=1)

    parser.add_option('-v', '--CONVERT', type=int, help="CONVERT PROJECT TO NUMPY", default=1)

    parser.add_option('-g', '--TARGET', type=int, help="NEED TO PARSING TARGETS", default=1)

    parser.add_option('-a', '--CREATE_DATASET', type=int, help="CREATE_DATASET", default=1)

    parser.add_option('-z', '--ADD_STATISTIC', type=int, help="ADD_STATISTIC to dataset 3 month", default=0)

    options, args = parser.parse_args()
    NEED_LOAD_FROM_DB = getattr(options, 'NEED_LOAD_FROM_DB')

    PATH_TO_TARGETS_EXCEL = getattr(options, 'PATH_TO_TARGETS_EXCEL')
    PATH_TO_SAVE_TARGETS = getattr(options, 'PATH_TO_SAVE_TARGETS')
    PATH_TO_PROJECTS = getattr(options, 'PATH_TO_PROJECTS')
    PATH_EXCEL_PROJECTS = getattr(options, 'PATH_EXCEL_PROJECTS')
    PATH_NPY_PROJECTS = getattr(options, 'PATH_NPY_PROJECTS')
    SAVE_WEIGHT = getattr(options, 'SAVE_WEIGHT')
    DOP_DATA_PATH = getattr(options, 'DOP_DATA_PATH')

    CONNECT = getattr(options, 'CONNECT')
    PROJ_ID = getattr(options, 'PROJ_ID')
    RELOAD_DOPS = getattr(options, 'RELOAD_DOPS')
    CONVERT = getattr(options, 'CONVERT')
    TARGET = getattr(options, 'TARGET')
    CREATE_DATASET = getattr(options, 'CREATE_DATASET')
    ADD_STATISTIC = getattr(options, 'ADD_STATISTIC')
    TRAIN: int = 0
    BATCH_SIZE: int = 8
    LR: float = 0.001
    EPOCH: int = 10
    NW: int = 0
    L2: float = 0.0001
    flag = 0
    error_flag = 0

    model_type = ModelType.Linear
    criteria_type = CriteriaType.MSE

    for dir_ in [PATH_TO_SAVE_TARGETS, PATH_TO_PROJECTS, PATH_EXCEL_PROJECTS, PATH_NPY_PROJECTS,
                 SAVE_WEIGHT, DOP_DATA_PATH, PATH_TO_TARGETS_EXCEL]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    if not os.path.exists(DOP_DATA_PATH + 'stages.xlsx'):
        create_stages(DOP_DATA_PATH)
        print_i(f'в папке {DOP_DATA_PATH} создался файл stages.xlsx')
        # dict_ = {'id': [], 'project_name': [], 'id_project': [], 'name_id': [], 'Note': []}
        # pd.DataFrame(dict_).set_index('id').to_excel(DOP_DATA_PATH + 'stages.xlsx')
        # print_e(f'!!! в папке{DOP_DATA_PATH}, нет stages.xlsx. Файл был создан пустым. !!! ')

    # PATH_TO_TARGETS_EXCEL = 'data/targets_excel/*.xlsx'
    # PATH_TO_SAVE_TARGETS = 'data'
    # PATH_EXCEL_PROJECTS = 'data/features/'
    # PATH_NPY_PROJECTS = 'data/prepred_train_data/'
    # SAVE_WEIGHT = 'data/WEIGHTS/'
    # DOP_DATA_PATH = 'data/dop_materials/'
    # NEED_LOAD_FROM_DB: int = 1
    # CONNECT: int = 1
    # PROJ_ID: int = 45700
    # RELOAD_DOPS: int = 0
    # CONVERT: int = 0
    # TARGET: int = 0
    # TRAIN: int = 0
    # CREATE_DATASET: int = 0
    # BATCH_SIZE: int = 8
    # region LOAD data from db
    if NEED_LOAD_FROM_DB:
        # 0 CONNECTION to DB
        # region Connect to db
        if CONNECT:
            print_i(f'CONNECT TO DB, PROJ_ID = {PROJ_ID}')
            df = pd.read_sql_query(sql_quary(proj_id=PROJ_ID), cnxn)
            # save to exel
            if not df.empty:
                flag = 1
                df.to_excel(PATH_EXCEL_PROJECTS + list(df['project_name'].unique())[0] + '.xlsx')
                print_i(
                    f"SUCCESS SAVE UNLOADING TO {PATH_EXCEL_PROJECTS + list(df['project_name'].unique())[0] + '.xlsx'}")
            else:
                print_e(f"DATA WITH PROJECT_ID = {PROJ_ID} IS EMPTY!")
                flag = 0
        # endregion

        # region RELOAD CONTRACTOR.npy MECH.npy STAGE.npy
        # 1 CREATE CONTRACTOR.npy MECH.npy STAGE.npy
        if RELOAD_DOPS:
            print_i(f'RELOAD: mech_res_dict, stages_dict')
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

            feature_contractors = []
            for i, path in enumerate(glob(PATH_EXCEL_PROJECTS + '*.xlsx')):
                print_d(i, path)
                df = pd.read_excel(path)
                feature_contractors.extend(pd.unique(df.contractor_name))

            feature_contractors = list(pd.unique(feature_contractors))
            feature_contractor_dict = dict(zip(feature_contractors, range(len(feature_contractors))))
            np.save(DOP_DATA_PATH + 'feature_contractor_dict.npy', feature_contractor_dict)
        # endregion

        # region convert projects to npy
        # 2 CONVERT to proj_data(len = 378)
        if CONVERT:
            print_i(f'CONVERT XLSX PROJECTS TO NPY')
            assert len([name for name in os.listdir(DOP_DATA_PATH) if
                        os.path.isfile(os.path.join(DOP_DATA_PATH, name))]) > 1, print_e(
                "Закинь данные в папку для доп материала")
            # target_contractors_dict = np.load(DOP_DATA_PATH + 'feature_contractor_dict.npy', allow_pickle=True).item()
            feature_contractor_dict = np.load(DOP_DATA_PATH + 'feature_contractor_dict.npy', allow_pickle=True).item()
            mech_res_dict = np.load(DOP_DATA_PATH + 'mech_res_dict.npy', allow_pickle=True).item()
            stages_dict = np.load(DOP_DATA_PATH + 'stages_dict.npy', allow_pickle=True).item()
            if CONNECT and flag:
                df['dt'] = df.dt.apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
                df['year'] = df.dt.dt.year
                df['month'] = df.dt.dt.month
                df['res_id'] = df.resource_name.map(mech_res_dict)
                df['contr_id'] = df.contractor_name.map(feature_contractor_dict)
                prepare_features(df, PATH_EXCEL_PROJECTS + list(df['project_name'].unique())[0] + '.xlsx',
                                 save_path=PATH_NPY_PROJECTS, stages_dict=stages_dict)
            else:
                for i, path in enumerate(glob(PATH_EXCEL_PROJECTS + '*.xlsx')):
                    print(i, path)
                    df = pd.read_excel(path)  # noqa
                    df['dt'] = df.dt.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                    df['year'] = df.dt.dt.year
                    df['month'] = df.dt.dt.month
                    df['res_id'] = df.resource_name.map(mech_res_dict)
                    df['contr_id'] = df.contractor_name.map(feature_contractor_dict)
                    prepare_features(df, path, save_path=PATH_NPY_PROJECTS, stages_dict=stages_dict)
            print_i('SUCCESS CONVERT PROJECTS TO NPY')
        # endregion
    # endregion

    # region NEED CREATE TARGET
    if TARGET:
        print_i('TRY TO CREATE TARGET')
        if len([name for name in os.listdir(PATH_TO_TARGETS_EXCEL) if os.path.isfile(os.path.join(PATH_TO_TARGETS_EXCEL, name))]) == 0:
            print_e(f"THERE IS NOT ANY FILE IN {PATH_TO_TARGETS_EXCEL}")
            error_flag = 1
        else:
            a = pd.DataFrame()
            year = 0
            for i, path_ in enumerate(glob(PATH_TO_TARGETS_EXCEL+'*.xls')):
                print_d(i, path_)
                dataframe = pd.read_excel(path_)  # noqa
                if os.path.splitext(path_)[-1] == '.xls':
                    dat = parse_data_xls(dataframe, fact=6)
                else:
                    dat = parse_data(dataframe)
                # dat = parse_data_xls(dataframe)

                year = int(dataframe.iloc[0, 2][14:18])
                month = int(dataframe.iloc[0, 2][11:13])

                # pd.DataFrame(dat).to_excel(f'{PATH_TO_SAVE_TARGETS}/{month}_{year}.xlsx')
                if i == 0:
                    a = pd.DataFrame(dat)
                a = pd.concat([a, pd.DataFrame(dat)], axis=0, join='outer', ignore_index=True)
            a.to_excel(f'{PATH_TO_SAVE_TARGETS}/whole_{year}.xlsx')
            # targets = pd.read_excel(f'{PATH_TO_SAVE_TARGETS}/whole_{year}.xlsx')  # noqa
            convert_target_to_npy(f'{PATH_TO_SAVE_TARGETS}/whole_{year}.xlsx', DOP_DATA_PATH, PATH_TO_SAVE_TARGETS)
            print_i(f"SUCCESS CREATE TARGET.NPY IN {PATH_TO_SAVE_TARGETS}")
    # else:
    #     print_i(f'TAKE TARGET FROM {PATH_TO_SAVE_TARGETS}')
        # targets = np.load(f'{PATH_TO_SAVE_TARGETS}/target_array.npy')
        # targets = np.load(f'{PATH_TO_SAVE_TARGETS}')
    # endregion

    # region Create DATASET
    if CREATE_DATASET:
        print_i('TRY TO CREATE DATASET')
        if error_flag:
            print_e(f'THERE IS NO target_array.npy IN {PATH_TO_SAVE_TARGETS}')
        else:
            targets = np.load(PATH_TO_SAVE_TARGETS + '/target_array.npy')
            print_i(f'TAKE TARGET FROM {PATH_TO_SAVE_TARGETS}')
            PD_tar = pd.DataFrame(targets)  # 0/1-contr/proj, 2-month, 3- year, 4 - res, 5 - val
            if PD_tar.empty:
                error_flag = 1

        Projects = list(glob(PATH_NPY_PROJECTS + '/*.npy'))[0:]
        if len(Projects) > 0 and not error_flag:
            dict_data = create_dataset(Projects, PD_tar)
            PD_DATA = pd.DataFrame(dict_data)
            PD_DATA.to_excel(PATH_TO_PROJECTS + 'DATA.xlsx')
            if ADD_STATISTIC:
                print_i('ADD 3 MONTH STATISTIC')
                Stat_PD_DATA = add_statistic(PD_DATA)
                Stat_PD_DATA.to_excel(PATH_TO_PROJECTS + 'Stat_PD_DATA.xlsx')
        else:
            if error_flag:
                print_e(f"ERROR WITH TARGETS WHEN TRY TO CREATE DATASET, BECAUSE error_flag = {error_flag} ")
            else:
                print_e(f"ERROR WHEN TRY TO CREATE DATASET, BECAUSE THERE ARE NO FILES IN PATH_NPY_PROJECTS")
                error_flag = 1
            #     Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS+ 'DATA.xlsx')  # noqa
    else:
        if os.path.exists(f'{PATH_TO_PROJECTS}+DATA.xlsx'):
            print_i(f'TAKE DATASET FROM {PATH_TO_PROJECTS}DATA.xlsx')
        else:
            print_e(f"THERE ARE NO ANY FILES IN {PATH_TO_PROJECTS}DATA.xlsx ")
            error_flag = 1

    if not error_flag:
        Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS + 'DATA.xlsx')  # noqa
        mech_res_dict = np.load(DOP_DATA_PATH + 'mech_res_dict.npy', allow_pickle=True).item()
        test_PD_DATA = Stat_PD_DATA.loc[(Stat_PD_DATA['month'] == 7) | (Stat_PD_DATA['month'] == 8)]
        test_PD_DATA = test_PD_DATA.loc[test_PD_DATA['year'] == 1]
        train_PD_DATA = Stat_PD_DATA[~Stat_PD_DATA.index.isin(test_PD_DATA.index)]
        train_PD_DATA = train_PD_DATA.sort_values(['year', 'month'])
        test_PD_DATA = test_PD_DATA.sort_values(by=['year'])

        # for lstm
        print_i(f'ModelType = {model_type}')
        if model_type is ModelType.LSTM:
            # train_PD_DATA = train_PD_DATA.sort_values(by=['year'] and ['month'])
            train_dataset = PROJDataset_sequenses(train_PD_DATA, mech_res_dict)
            test_dataset = PROJDataset_sequenses(test_PD_DATA, mech_res_dict)
        else:
            # for linear model
            train_dataset = PROJDataset(train_PD_DATA)
            test_dataset = PROJDataset(test_PD_DATA)

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # for i, d in enumerate(train_loader):
        #     print(i, d[0].shape, d[1].shape)
        #     print(f'data = {d[0]}')
        #     print(f'targets = {d[1]}')

    # endregion

    # region TRAIN model
    if TRAIN:
        config = {
            "learning_rate": LR,
            "epochs": EPOCH,
            "batch_size": BATCH_SIZE,
            "num_workers": NW,
            "weight_decay(l2)": L2,
        }
        parametrs = Parameters(config, model_type, criteria_type)
        train_model(parametrs.net, train_loader, test_loader, parametrs.criteria, 0, parametrs.optimizer, 0,
                    parametrs.epochs, SAVE_WEIGHT, 'name')
        print(f"Done. Model saved in folder [{SAVE_WEIGHT}]")
    # endregion
