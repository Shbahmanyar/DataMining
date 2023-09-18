import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('insurance_claims.csv' )
df['police_report_available']=df['police_report_available'].replace({'?':np.nan})
df['police_report_available']=df['police_report_available'].fillna(method='ffill')
df['collision_type']=df['collision_type'].replace({'?':np.nan})
df['collision_type']=df['collision_type'].fillna(method='ffill')
df['property_damage']=df['property_damage'].replace({'?':np.nan})
df['property_damage']=df['property_damage'].fillna(method='ffill')
df=df.drop(['months_as_customer','policy_number','policy_bind_date','policy_csl','auto_year','auto_model','insured_hobbies','insured_zip'],axis=1)

le=LabelEncoder()
for i in list(df.columns):
    if df[i].dtype=='object':
        df[i]=le.fit_transform(df[i])
        
X=df.drop('fraud_reported', axis=1) 
y=df['fraud_reported']

df = X
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=y,cmap='rainbow')
plt.xlabel('First component')
plt.ylabel('Second Component')
plt.show()
