import pandas as pd
from pandasql import sqldf

df = pd.read_csv('T.csv')
sql ('SELECT *FROM df ' , globals())
sqldf ('SELECT ParentId , CountyName,AVG("CarPrice_Sum") FROM df GROUP BY ParentId,CountyName' ,globals())

