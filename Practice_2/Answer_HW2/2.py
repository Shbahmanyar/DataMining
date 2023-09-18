import pandas as pd
df = pd.read_csv('T.csv')
data = df[df["ProvinceName"] == "تهران"]
data = data[data["IsBimarkhas"] == 1]
data = data[["ProvinceName","IsBehzisti_Malool","CountyName"]]
d=data.groupby(["CountyName","IsBehzisti_Malool"])["ProvinceName"].count()
print(d)
