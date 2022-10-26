import pandas as pd
import numpy as np
import openpyxl
from openpyxl import load_workbook


def convert_target_to_npy(target_xlsx: str = None, path_to_dop_materials: str = None, path_to_save: str = None) -> None:
    targets = pd.DataFrame(pd.read_excel(f'{target_xlsx}'))
    # target_contractors_dict = df_contractors.dropna().set_index('target_contractors').id.to_dict()
    targets_contractors = list(pd.unique(targets.contractor))
    target_contractors_dict = dict(zip(targets_contractors, range(len(targets_contractors))))
    np.save(path_to_dop_materials + 'targets_contractors_dict.npy', target_contractors_dict)

    feature_contractor_dict = np.load(path_to_dop_materials + 'feature_contractor_dict.npy', allow_pickle=True).item()
    mech_res_dict = np.load(path_to_dop_materials + 'mech_res_dict.npy', allow_pickle=True).item()

    stages_dict = np.load(path_to_dop_materials + 'stages.npy', allow_pickle=True)
    targets['res_id'] = targets.res_name.map(mech_res_dict)
    targets['contr_id'] = targets.contractor.map(target_contractors_dict)

    mech_missed_ids_dict = pd.read_excel(path_to_dop_materials+'mech_missed_ids.xlsx', usecols=[0, 1]).dropna()\
        .set_index('missed_name').id.to_dict()
    mech_res_dict_updated = {**mech_missed_ids_dict, **mech_res_dict}
    targets['res_id'] = targets.res_name.map(mech_res_dict_updated)

    targets['proj_id'] = targets.stage.map(stages_dict.item())
    target_df = targets.loc[:, ['proj_id', 'contr_id', 'month', 'year', 'res_id', 'hours']]
    target_df.dropna(axis='index', inplace=True)
    target_array = target_df.to_numpy(dtype=float)
    np.save(f'{path_to_save}/target_array.npy', target_array, allow_pickle=True)



