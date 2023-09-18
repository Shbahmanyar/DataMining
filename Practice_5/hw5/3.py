import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
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

y = label_binarize(y, classes=[0,1,2,3 ,4])
n_classes = y.shape[1]
print(n_classes)
random_state = np.random.RandomState(42)
n_samples, n_features = X.shape


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf',max_iter=200, probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

f = dict()
t = dict()
roc_auc = dict()
for i in range(n_classes):
    f[i], t[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(f[i], t[i])


f["micro"], t["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(f["micro"], t["micro"])

plt.figure()
lw = 2
plt.plot(f[2], t[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


all_f = np.unique(np.concatenate([f[i] for i in range(n_classes)]))


mean_t = np.zeros_like(all_f)
for i in range(n_classes):
    mean_t += interp(all_f, f[i], t[i])


mean_t /= n_classes

f["macro"] = all_f
t["macro"] = mean_t
roc_auc["macro"] = auc(f["macro"], t["macro"])


plt.figure()
plt.plot(f["micro"], t["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(f["macro"], t["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(f[i], t[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
print("end")


