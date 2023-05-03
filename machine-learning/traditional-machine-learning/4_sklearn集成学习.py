from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier,BaggingClassifier
import warnings
warnings.filterwarnings("ignore")

#数据加载
iris = load_iris()
data = iris['data']
target = iris['target']
#数据划分
x_train,x_val,y_train,y_val = train_test_split(data,target,train_size=0.7)

#模型构建
cls1 = LogisticRegression()
cls2 = DecisionTreeClassifier()
cls3 = SGDClassifier()
clss = {'LR':cls1,'DT':cls2,'SGD':cls3}
for k,v in clss.items():
    model = v
    model.fit(x_train,y_train)
    y_pred = model.predict(x_val)
    print('%s结果:'%k, classification_report(y_val, y_pred))


# model = VotingClassifier(
#     # estimators=[('LR', cls1), ('DT', cls2), ('SGD', cls3)],
#     estimators=[(i,v) for i,v in clss.items()],
#     voting='hard')

model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10)

#模型训练
model.fit(x_train,y_train)
#模型预测
y_pred = model.predict(x_val)
#指标评价
print('集成学习结果:',classification_report(y_val,y_pred))
