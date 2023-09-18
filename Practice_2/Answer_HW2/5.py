'''import pandas as pd
from pandasql import sqldf

df = pd.read_csv('T.csv')
sql ('SELECT *FROM df ' , globals())
sqldf ('SELECT ParentId , CountyName,AVG("CarPrice_Sum") FROM df GROUP BY ParentId,CountyName' ,globals())

'''

import pandas as pd
df = pd.read_csv('T.csv')
data=df.groupby(['ParentId'])['Id'].count()
data1=data.groupby(['ParentId'])['Average_price'].sum()
d=pd.concat([data , data1])
print(d)
