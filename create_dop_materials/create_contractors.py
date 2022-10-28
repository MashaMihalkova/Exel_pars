import pandas as pd
import numpy as np
from sql_connection.queary_to_bd import *


def create_contractors(path_to_dop_materials: str = None):
    sql_ = """
                SELECT  DISTINCT (uca.contractor_name) 
                from publication.UDF_CODE_Activity uca
                where uca.contractor_name is not NULL 
           """
    df = pd.read_sql_query(sql_, cnxn)
    contractors_dict = dict(zip(df['contractor_name'], range(len(df))))
    pd.DataFrame(contractors_dict.items()).to_excel(path_to_dop_materials + 'all_contractors.xlsx')
    np.save(path_to_dop_materials + 'all_contractors.npy', contractors_dict, allow_pickle=True)


