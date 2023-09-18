#میانگین سفرهاس غیر زیارتی استان اصفهان در سال 98 به تفیکیک شهر
import pandas as pd
df = pd.read_csv('T.csv')
data= df[df['ProvinceName'] == "اصفهان"]
data=data.groupby (['CountyName'])[['Trip_AirNonPilgrimageCount_98']].mean()
print(data)
