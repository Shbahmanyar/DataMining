import pandas as pd
df = pd.read_csv('T.csv')
data=df.groupby(['ParentId'])['Id'].count()
data1=data.groupby(['ParentId'])['Average_price'].sum()
d=pd.concat([data , data1])
print(d)
