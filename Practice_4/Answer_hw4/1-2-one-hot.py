import pandas as pd
from pathlib import Path


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
