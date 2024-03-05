# decisiontree
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pydot
df=pd.read_csv('/content/kyphosis.csv')
X=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions=dtree.predict(X_test)
print("Accuracy Score",accuracy_score(y_test,predictions))
features=list(df.columns[1:])
print(features)
dot_data=StringIO()

export_graphviz(dtree,out_file=dot_data,feature_names=features,filled=True,rounded=True)
graph=pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())
