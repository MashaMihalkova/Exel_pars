
import os
import pandas as pd
from databese import SessionLocal
from pars_terget import *
from extract_data_from_bd import sql_quary
from sqlalchemy.sql import text
import psycopg2
from create_database import *
if __name__ == '__main__':
    Path_to_file = 'D:\\work2\\S-krivaia\\data_sila_sibir\\sila_sibir_july.xlsx'
    Path_to_save = 'D:\\work2\\S-krivaia\\data_sila_sibir\\'

    # data = pars(Path_to_file)
    # df = pd.DataFrame(data)
    # df.to_csv(os.path.join(Path_to_save, 'targets_pars_july_ooo.csv'))
    # df.to_excel(os.path.join(Path_to_save, 'targets_pars_july_ooo.xlsx'))
    # connection = psycopg2.connect(host='sql2019d01.cs.local', port='1433',  user='PMAdmin', password='PMAdmin',
    #                               database='PMControlling')
    #
    # cursor = connection.cursor()
    # quary_ = pd.read_sql_query(sql_quary, connection)
    # print(1)
    # with SessionLocal() as db:
    #     # objects = db.query(sql_quary).all()
    #     # ob = db.query('''select * from publication.resource''')
    #     od = db.text('''select * from publication.resource''')
    #     print(1)

