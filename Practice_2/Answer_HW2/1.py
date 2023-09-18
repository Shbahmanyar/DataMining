import pandas as pd
data = pd.read_csv('T.csv')
data=data.groupby (['CountyName'])[['Daramad_Total_Rials']].mean()
print(data)
