from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']

from sklearn.preprocessing import StandardScaler,MinMaxScaler

housing_boston = load_boston()
x = housing_boston['data']     # datas
y = housing_boston['target']   # label
x = StandardScaler().fit_transform(x)


x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.3)
model = RandomForestRegressor()
model.fit(x_train,y_train)

y_pred = model.predict(x_val)

print(r2_score(y_val,y_pred))

plt.figure()
plt.plot(y_val,label='val')
plt.plot(y_pred,label='pred')
plt.legend()
plt.title('boston房价预测,R2=%.3f'%r2_score(y_val,y_pred))
plt.show()


