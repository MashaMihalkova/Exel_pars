import pyodbc
import pandas as pd
server = 'sql2019d01.cs.local,1433'
database = 'ASU_GSP_dev'
username = 'PMAdmin'
password = 'PMAdmin'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};Server='
                      +server+';Database='+database+';uid='+username+
                      ';pwd='+ password+';TrustServerCertificate=yes;Encrypt=yes')
cursor = cnxn.cursor()

server_target = '10.0.1.64,5432'
database_target = 'db_track_torch'
username_target = 'osm'
password_target = 'geoserver'
cnxn_target = pyodbc.connect('DRIVER={PostgreSQL Unicode};Server='
                      +server_target+';Database='+database_target+';uid='+username_target+
                      ';pwd='+ password_target+';TrustServerCertificate=yes;Encrypt=yes')
cursor_target = cnxn_target.cursor()

#
# import csv
#
# sql = """
# SELECT * FROM publication.RESOURCE
# """
# rows = cursor.execute(sql)
# with open(r'.\test.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow([x[0] for x in cursor.description])  # column headers
#     for row in rows:
#         writer.writerow(row)