import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix,classification_report, confusion_matrix

df = pd.read_csv('insurance_claims.csv')
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
    
X=df.drop('fraud_reported', axis=1) #hame joz fraud_reported
y=df['fraud_reported']

X_train, X_test,y_train ,y_test= train_test_split(X,y, random_state = 42)
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)
y_train_scaled = preprocessing.scale(y_train)
y_test_scaled = preprocessing.scale(y_test)

X_train, X_test,y_train ,y_test= train_test_split(X,y,train_size=.70, test_size = 0.30, random_state = 42)
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)
y_train_scaled = preprocessing.scale(y_train)
y_test_scaled = preprocessing.scale(y_test)

param_grid = {'C': [1, 5, 10,100],
                      'gamma': [1, 0.1,0.01 ,5],
                      'kernel': ['linear','rbf','poly','sigmoid']}
grid = GridSearchCV(SVC(max_iter=200), param_grid, refit=True, verbose=3, n_jobs=-1)
grid.fit(X_train, y_train)
print("grid.best_params_")
print(grid.best_params_)
grid_predictions = grid.predict(X_test)


svclassifier = SVC(kernel='rbf', max_iter=200 , C=1, gamma=1).fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
opt = [("Confusion matrix, without normalization for linear", None),
                  ("Normalized confusion matrix for linear", 'true')]
for title, normalize in opt:
    disp = plot_confusion_matrix(svclassifier, X_test, y_test, normalize=normalize ,cmap=plt.cm.Blue)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


