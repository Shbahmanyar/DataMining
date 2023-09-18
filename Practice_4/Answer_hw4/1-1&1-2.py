from sklearn import tree
import pandas as pd
from sklearn import preprocessing
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pd.set_option('max_columns', None)

df = pd.read_csv('processed.cleveland.csv')
df.columns=['age' ,'sex' ,'cp ' ,'restbp' ,'chol' ,'fbs' ,'restecg' ,'thalach' ,'exang' ,'oldpeak' ,'slope' ,'ca' ,'thal' ,'hd']
df=df[(df !='?').all(axis=1)]
X=df.drop('hd', axis=1)
y=df['hd']


file = Path('processed.cleveland.data').read_text()
s = list()
for m in file.split('\n'):
    mylist = m.split(',')
    if("?"in mylist):continue
    if(""in mylist):continue
    s.append(mylist)
attribute=['age' ,'sex' ,'cp' ,'restbp' ,'chol' ,'fbs' ,'restecg' ,'thalach' ,'exang' ,'oldpeak' ,'slope' ,'ca' ,'thal' ,'hd']
df = pd.DataFrame(s, columns= attribute)
attribute1 = [ 'cp',  'restecg','slope', 'thal']
for mead in attribute1:
    print(mead," :\n",pd.get_dummies(df[mead]))

##print(pd.get_dummies(df['restecg']))
##print(pd.get_dummies(df['slope']))
##print(pd.get_dummies(df['thal']))
##print(pd.get_dummies(df['cp']))

X_train, X_test,y_train ,y_test= train_test_split(X,y,train_size=.70, random_state = 42)

X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)
y_train_scaled = preprocessing.scale(y_train)
y_test_scaled = preprocessing.scale(y_test)
#print(y_train_scaled)
#print(x_train_scaled)

plt.figure(figsize=(5, 5))
X_train,X_test,y_train,y_test=train_test_split(X, y, random_state=42)
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
tree.plot_tree(clf)
plt.show()
##m=['cp', 'restcg']
##for s in m :
##    print (s , ':\n' , pd.get_dummies(data[s]))

##df = pd.DataFrame(columns= data.columns)
##pd.set_option("display.max_columns", 99)
##pd.set_option("display.max_rows", 999)
##myfields2 = [ 'cp', 'restecg','slope', 'thal']
##
##for field in myfields2:
##     print(field," :\n",pd.get_dummies(df[field]))

#print(pd.get_dummies(df ,df.columns=['cp']))

#print(pd.get_dummies(pd.series('cp') ))
