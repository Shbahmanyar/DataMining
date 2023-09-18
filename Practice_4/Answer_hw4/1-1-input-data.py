import pandas as pd

pd.set_option('max_columns', None)
data = pd.read_csv('processed.cleveland.csv')
data.columns=['age' ,'sex' ,'cp ' ,'restbp' ,'chol' ,'fbs' ,'restecg' ,'thalach' ,'exang' ,'oldpeak' ,'slope' ,'ca' ,'thal' ,'hd']
data=data[(data !='?').all(axis=1)]
print(data)
X=data.drop('hd', axis=1) 
#print(X)
y=data['hd'] 
#print(y)    


