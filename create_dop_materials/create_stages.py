import pandas as pd
import numpy as np
from sql_connection.queary_to_bd import *


def create_stages(path_to_dop_materials: str = None):

    projects_GSP = pd.DataFrame(pd.read_excel(path_to_dop_materials + 'Проекты ГСП.xls'))

    sql = '''
           select p.id_project, p.project_code from publication.PROJECT as p
          '''
    df = pd.read_sql_query(sql, cnxn)
    id_new = 0
    dict_ = {'id': [], 'project_name': [], 'id_project': [], 'name_id': [], 'Note': []}
    for proj_code in np.unique(projects_GSP['ID проекта P6'].loc[projects_GSP['ID проекта P6'].notna()]):
        find_proj = df.loc[df.project_code == proj_code]
        stage_name = projects_GSP.loc[projects_GSP['ID проекта P6'] == proj_code]
        if not find_proj.empty:
            stage = list(stage_name['Проект'])[0]
            id_project = list(find_proj.id_project)[0]
            dict_['id'].append(id_new)
            dict_['project_name'].append(stage)
            dict_['id_project'].append(id_project)
            dict_['name_id'].append(proj_code)
            dict_['Note'].append('Выгружен')
            id_new += 1

    pd_stage = pd.DataFrame(dict_)
    pd_stage.to_excel(path_to_dop_materials + 'stages.xlsx')
    stages_dict = dict(zip(list(dict_['project_name']), list(dict_['id'])))
    stages = pd_stage.to_numpy()
    np.save(path_to_dop_materials + 'stages.npy', stages_dict, allow_pickle=True)
