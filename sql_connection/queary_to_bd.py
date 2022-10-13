import pyodbc
import pandas as pd
server = 'sql2019d01.cs.local,1433'
database = 'ASU_GSP_dev'
username = 'PMAdmin'
password = 'PMAdmin'
cnxn = pyodbc.connect('DRIVER={SQL Server};Server='
                      +server+';Database='+database+';uid='+username+
                      ';pwd='+ password+';TrustServerCertificate=yes;Encrypt=yes')
cursor = cnxn.cursor()




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