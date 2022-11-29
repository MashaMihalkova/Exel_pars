import pandas as pd
import numpy as np


def status_tech(path_to_status: str, path_to_id_interobj: str, path_to_save: str) -> None:
    Data_status = pd.read_excel(path_to_status)  # noqa
    Data_id = pd.read_excel(path_to_id_interobj)  # noqa
    Data_stages = pd.read_excel('data/dop_materials/stages.xlsx')  # noqa
    Data_status['Status'].unique()
    Data_Res = pd.DataFrame()
    for proj_name in Data_stages['project_name'].unique():
        id_proj = Data_id.loc[Data_id['NMINTROBJ'] == proj_name, 'IDINTROBJ'].values[0]
        d = Data_status.loc[Data_status['IDINTROBJ'] == id_proj]
        if not d.empty():
            dd = d.replace({'IDINTROBJ': id_proj}, proj_name)
            Data_Res = pd.concat((Data_Res, dd))
    Data_Res.to_excel(path_to_save)
    print(1)




