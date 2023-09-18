
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn import tree
import sklearn

pd.set_option('max_columns', None)
data = pd.read_csv('processed.cleveland.csv')
data.columns=['age' ,'sex' ,'cp ' ,'restbp' ,'chol' ,'fbs' ,'restecg' ,'thalach' ,'exang' ,'oldpeak' ,'slope' ,'ca' ,'thal' ,'hd']
data=data[(data !='?').all(axis=1)]
X=data.drop('hd', axis=1)
y=data['hd']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30,random_state=42)
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(random_state=42)
tree.fit(X_train,y_train)
plt.figure(figsize=(25, 25))
sklearn.tree.plot_tree(tree)
plt.show()


y_train_pred=tree.predict(X_train)
y_test_pred=tree.predict(X_test)

from sklearn.metrics import  accuracy_score
print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))
path=tree.cost_complexity_pruning_path(X_train,y_train)
alphas=path['ccp_alphas']

accuracy_train,accuracy_test=[],[]
cv_list=[]
cv_std=[]
cv_mean=[]
cv_scores=[]

for i in alphas:
    tree=DecisionTreeClassifier(ccp_alpha=i)

    tree.fit(X_train,y_train)
    
    sklearn.tree.plot_tree(tree)
    plt.show()
    cv_scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy')
    cv_list.append(cv_scores)
    cv_mean.append(cv_scores.mean())
    cv_std.append(cv_scores.std())
    y_train_pred=tree.predict(X_train)
    y_test_pred=tree.predict(X_test)
    accuracy_train.append(accuracy_score(y_train,y_train_pred))
    accuracy_test.append(accuracy_score(y_test,y_test_pred))
sns.set()
plt.figure(figsize=(5,5))
sns.lineplot(y=accuracy_train,x=alphas,label="Train accuracy")
sns.lineplot(y=accuracy_test,x=alphas,label="test accuracy")
plt.xticks(ticks=np.arange(0.00,0.25,0.01))
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(alphas, cv_mean, marker='o', label="mean", drawstyle="steps-post")
ax.plot(alphas, cv_std, marker='o', label="std",drawstyle="steps-post")
ax.legend()
plt.show()
print("cv_list ",cv_list)
print("cv_mean ", cv_mean)
print("cv_std ", cv_std)

tree = DecisionTreeClassifier(ccp_alpha=0.022,random_state=42)
tree.fit(X_train,y_train)
sklearn.tree.plot_tree(tree)
plt.show()

y_train_pred=tree.predict(X_train)
y_test_pred=tree.predict(X_test)
print(accuracy_score(y_train,y_train_pred),accuracy_score(y_test,y_test_pred))

