#میانگین در امد افراد استان اصفهان به تفکیک شهر
import pandas as pd
data = pd.read_csv('T.csv')
data= df[df['ProvinceName'] == "اصفهان"]
data=data.groupby (['CountyName'])[['Daramad_Total_Rials']].mean()
print(data)
