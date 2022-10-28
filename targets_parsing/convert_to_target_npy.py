import pandas as pd
import numpy as np
import openpyxl
from openpyxl import load_workbook
import re


def convert_target_to_npy(target_xlsx: str = None, path_to_dop_materials: str = None, path_to_save: str = None) -> None:
    targets = pd.DataFrame(pd.read_excel(f'{target_xlsx}'))
    # target_contractors_dict = df_contractors.dropna().set_index('target_contractors').id.to_dict()
    # targets_contractors = list(pd.unique(targets.contractor))
    # target_contractors_dict = dict(zip(targets_contractors, range(len(targets_contractors))))
    # np.save(path_to_dop_materials + 'targets_contractors_dict.npy', target_contractors_dict)

    # feature_contractor_dict = np.load(path_to_dop_materials + 'feature_contractor_dict.npy', allow_pickle=True).item()
    mech_res_dict = np.load(path_to_dop_materials + 'mech_res_dict.npy', allow_pickle=True).item()
    all_contractors_dict = np.load(path_to_dop_materials+'all_contractors.npy', allow_pickle=True).item()
    pd_all_contractors_dict = pd.DataFrame(all_contractors_dict.items())
    k = pd_all_contractors_dict[0]
    # u = [re.findall(r'\"(.{1,})\"', i)[0] if re.findall(r'\"(.{1,})\"', i) else i for i in list(k.values)]
    u = [re.findall(r'(.{1,})(\"(.{1,})\")', i)[0][1].replace("\"",'')+' '+re.findall(r'(.{1,})(\"(.{1,})\")', i)[0][0][:-1] if re.findall(r'\"(.{1,})\"', i) else i for i in list(k.values)]
    pd_all_contractors_dict['similar_name'] = u
    pd_all_contractors_dict = pd_all_contractors_dict.drop(0, axis=1)
    # dict_all_contractor = pd_all_contractors_dict.to_dict('series')
    # dict_all_contractor_ = pd_all_contractors_dict.to_dict()
    w = [list(pd_all_contractors_dict.values[i]) for i in range(pd_all_contractors_dict.shape[0])]
    w_np = [x[::-1] for x in w]
    similar_contractors_dict = dict(w_np)
    similar_contractors_dict_lower = {k.lower(): v for k, v in similar_contractors_dict.items()}
    all_contractors_dict_lower = {k.lower(): v for k, v in all_contractors_dict.items()}


    stages_dict = np.load(path_to_dop_materials + 'stages.npy', allow_pickle=True)
    targets['res_id'] = targets.res_name.map(mech_res_dict)
    # targets['contr_id'] = targets.contractor.map(target_contractors_dict)
    targets['contr_id'] = targets.contractor.str.lower().map(all_contractors_dict_lower)
    # targets['contr_id'] = targets.loc[targets['contr_id_'].isna]
    targets['contr_id'] = targets.contr_id.fillna(targets.contractor.str.lower().map(similar_contractors_dict_lower))

    mech_missed_ids_dict = pd.read_excel('data/Needed_materials/mech_missed_ids.xlsx', usecols=[0, 1]).dropna()\
        .set_index('missed_name').id.to_dict()
    mech_res_dict_updated = {**mech_missed_ids_dict, **mech_res_dict}
    targets['res_id'] = targets.res_name.map(mech_res_dict_updated)

    targets['proj_id'] = targets.stage.map(stages_dict.item())
    target_df = targets.loc[:, ['proj_id', 'contr_id', 'month', 'year', 'res_id', 'hours']]
    target_df.dropna(axis='index', inplace=True)
    target_array = target_df.to_numpy(dtype=float)
    np.save(f'{path_to_save}/target_array.npy', target_array, allow_pickle=True)



