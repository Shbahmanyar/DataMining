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
#df = df.replace('?', np.nan)
#result = df.head(10)
#print(result[['collision_type']])
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


print("linear : ")
svclassifier = SVC(kernel='linear',max_iter=200,gamma=1).fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
opt = [("Confusion matrix, without normalization for linear", None), ("Normalized confusion matrix for linear", 'true')]
for title, normalize in opt:
    disp = plot_confusion_matrix(svclassifier, X_test, y_test, cmap=plt.cm.Blues, normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()

print("rbf : ")
svclassifier = SVC(kernel='rbf', gamma=1).fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
opt = [("Confusion matrix, without normalization for rbf", None), ("Normalized confusion matrix for rbf", 'true')]
for title, normalize in opt:
    disp = plot_confusion_matrix(svclassifier, X_test, y_test,normalize=normalize ,cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


print("poly : ")
svclassifier = SVC(kernel='poly').fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
opt = [("Confusion matrix, without normalization for poly", None), ("Normalized confusion matrix for poly", 'true')]
for title, normalize in opt:
    disp = plot_confusion_matrix(svclassifier, X_test, y_test, normalize=normalize ,cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()



print("sigmoid : ")
svclassifier = SVC(kernel='sigmoid', C=1, gamma=1).fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
opt = [("Confusion matrix, without normalization for sigmoid", None), ("Normalized confusion matrix for sigmoid", 'true')]
for title, normalize in opt:
    disp = plot_confusion_matrix(svclassifier, X_test, y_test, cmap=plt.cm.Blues, normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()



##        parameters = {'C': [0.1, 5, 10 ,100],
##                      'gamma': [1, 0.1, 0.01 ,0.001 ,0.0001],
##                      'gamma': ['scale' ,'auto'],
##                      'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
##        clf = GridSearchCV(SVC(), parameters ,refit =true ,verbose=3, n_jobs=-1)
##        clf.fit(X_train, y_train)
##        print("grid.best_parameters_")
##        print(grid.best_parameters_)
##        grid_predictions = grid.predict(X_test)
##        print("classification_report(y_test, grid_predictions)")
##        print(classification_report(y_test, grid_predictions))

       
##
##        svclassifier = SVC(kernel='linear', max_iter=200 , C=1, gamma=5).fit(X_train, y_train)
##        y_pred = svclassifier.predict(X_test)
##        print(confusion_matrix(y_test, y_pred))
##        print(classification_report(y_test, y_pred))
##        opt = [("Confusion matrix, without normalization for linear", None),
##                          ("Normalized confusion matrix for linear", 'true')]
##        for title, normalize in opt:
##            disp = plot_confusion_matrix(svclassifier, X_test, y_test, normalize=normalize ,cmap=plt.cm.Blue)
##            disp.ax_.set_title(title)
##            print(title)
##            print(disp.confusion_matrix)
##        plt.show()





