from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


#数据加载
iris = load_iris()
data = iris['data']
target = iris['target']

data_ = np.concatenate([data.reshape(-1,4),target.reshape(-1,1)],axis=1)
data_ = pd.DataFrame(data_)
data_.to_excel('1.xlsx')



#模型构建
model = LogisticRegression()
# model = RandomForestClassifier()
#数据划分
x_train,x_val,y_train,y_val = train_test_split(data,target,train_size=0.7)
#模型训练
model.fit(x_train,y_train)
#模型预测
y_pred = model.predict(x_val)

#指标评价
print(classification_report(y_val,y_pred))
