import pandas as pd
df = pd.read_csv('T.csv')
data= df[df['ProvinceName'] == "مازندران"]
data= data[data['IsUrban'] == 1]
data= data[ data['SenfName'].notnull()]
data["BirthDate"] = pd.to_datetime(data['BirthDate']).dt.year
data=data.groupby (['BirthDate' ,'CountyName'])["ProvinceName"].count()
print(data)

