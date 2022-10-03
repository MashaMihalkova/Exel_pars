import numpy as np

from targets_parsing.pars_targets import *
# from sql_connection.queary_to_bd import *
from sql_connection.extract_data_from_bd import sql_quary
from create_dataset.create_dataset_ import *
from glob import glob
from train_model import train_model
from wand_config.config import *
from datetime import datetime
from create_dataset.converter_from_database_loader_to_npy import *

# ПОДГОТОВКА ДАННЫХ
# 0. чтение из бд по ид_проекта
# 1. создать stage.npy contractor.npy и тд
# 2. преобразуем полученные данные в нужный вид (proj_id, contr_id, PO, month, year, res_id) len = 378
#     сохранение в npy (но мб можно и упустить это
# ПОДГОТОВКА ТАРГЕТА
# 3. Читаем таргет
# СОЗДАНИЕ БД
# 4. запускаем create_dataset


if __name__ == '__main__':
    PROJ_ID: int = 45700
    PATH_TO_TARGETS_EXCEL = 'data/targets_excel/*.xlsx'
    PATH_TO_SAVE_TARGETS = 'data'
    PATH_EXCEL_PROJECTS = 'data/features/'
    PATH_NPY_PROJECTS = 'data/prepred_train_data2/'
    SAVE_WEIGHT = 'data/WEIGHTS/'
    DOP_DATA_PATH = 'data/dop_materials/'
    NEED_LOAD_FROM_DB: int = 1
    CONNECT: int = 0
    RELOAD_DOPS: int = 0
    CONVERT: int = 0
    TARGET: int = 0
    TRAIN: int = 0
    if NEED_LOAD_FROM_DB:
        # 0 CONNECTION to DB
        # if CONNECT:
        #     df = pd.read_sql_query(sql_quary(proj_id=PROJ_ID), cnxn)
            # сохранить в exel

        # 1 CREATE CONTRACTOR.npy MECH.npy STAGE.npy
        if RELOAD_DOPS:
            df = pd.read_excel(DOP_DATA_PATH+'Norms.xlsx')
            mech_res_list = pd.unique(df.loc[pd.notna(df.iloc[:, 4]), 'Работа\специальность\техника'])
            mech_res_dict = dict(zip(mech_res_list, range(len(mech_res_list))))
            np.save(DOP_DATA_PATH+'mech_res_dict.npy', mech_res_dict, allow_pickle=True)

            df_stages = pd.read_excel(DOP_DATA_PATH+'/stages.xlsx')
            stages_dict = df_stages.set_index('project_name').id.to_dict()
            np.save(DOP_DATA_PATH+'stages_dict.npy', stages_dict, allow_pickle=True)

            # TODO: feature_contractor_dict

        # 2 CONVERT to proj_data(len = 378)
        if CONVERT:
            target_contractors_dict = np.load(DOP_DATA_PATH + 'feature_contractor_dict.npy', allow_pickle=True).item()
            feature_contractor_dict = np.load(DOP_DATA_PATH + 'feature_contractor_dict.npy', allow_pickle=True).item()
            mech_res_dict = np.load(DOP_DATA_PATH + 'mech_res_dict.npy', allow_pickle=True).item()
            stages_dict = np.load(DOP_DATA_PATH + 'stages_dict.npy', allow_pickle=True).item()

            for i, path in enumerate(glob(PATH_EXCEL_PROJECTS + '*.xlsx')):
                print(i, path)
                df = pd.read_excel(path)
                df['dt'] = df.dt.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                df['year'] = df.dt.dt.year
                df['month'] = df.dt.dt.month
                df['res_id'] = df.resource_name.map(mech_res_dict)
                df['contr_id'] = df.contractor_name.map(feature_contractor_dict)
                prepare_features(df, path, save_path=PATH_NPY_PROJECTS, stages_dict=stages_dict)

        # 3 TARGET
        if TARGET:
            for i, path_ in enumerate(glob(PATH_TO_TARGETS_EXCEL)):
                print(i, path_)
                dataframe = pd.read_excel(path_)
                dat = parse_data(dataframe)

                year = int(dataframe.iloc[0, 2][14:18])
                month = int(dataframe.iloc[0, 2][11:13])

                pd.DataFrame(dat).to_excel(f'{PATH_TO_SAVE_TARGETS}/{month}_{year}.xlsx')
                if i == 0:
                    a = pd.DataFrame(dat)
                a = pd.concat([a, pd.DataFrame(dat)], axis=0, join='outer', ignore_index=True)

            a.to_excel(f'{PATH_TO_SAVE_TARGETS}/whole_{year}.xlsx')
            targets = pd.read_excel(f'{PATH_TO_SAVE_TARGETS}/whole_{year}.xlsx')
        else:
            targets = np.load(f'{PATH_TO_SAVE_TARGETS}/target_array.npy')

    # 4 CREATE DATASET
    PD_tar = pd.DataFrame(targets)  # 0/1-contr/proj, 2-month, 3- year, 4 - res, 5 - val
    Projects = list(glob(PATH_NPY_PROJECTS + '/*.npy'))[1:]
    dict_data = create_dataset(Projects, PD_tar)
    PD_DATA = pd.DataFrame(dict_data)

    test_PD_DATA = PD_DATA.loc[(PD_DATA['month'] == 7) | (PD_DATA['month'] == 8)]
    test_PD_DATA = test_PD_DATA.loc[test_PD_DATA['year'] == 1]
    train_PD_DATA = PD_DATA[~PD_DATA.index.isin(test_PD_DATA.index)]

    train_dataset = PROJDataset(train_PD_DATA)
    test_dataset = PROJDataset(test_PD_DATA)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=wandb.config['batch_size'], shuffle=False)

    # 5 Train model
    if TRAIN:
        train_model(net, train_loader, test_loader, criteria, 0, optimizer, 0, epochs, SAVE_WEIGHT, 'name')
        print(f"Done. Model saved in folder [{SAVE_WEIGHT}]")
