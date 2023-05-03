from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
import time
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']


housing_boston = load_boston()
x = housing_boston['data']     # datas
y = housing_boston['target']   # label
x = StandardScaler().fit_transform(x)
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.3,random_state=1)

time1 = time.perf_counter()
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_val)
print('无PCA的R2:',r2_score(y_val,y_pred))
time2 = time.perf_counter()

pca = PCA(n_components=6)
pca.fit(x)
x_train_pca = pca.transform(x_train)
x_val_pca = pca.transform(x_val)

time3 = time.perf_counter()
model = LinearRegression()
model.fit(x_train_pca,y_train)
y_pred_pca = model.predict(x_val_pca)
print('有PCA的R2:',r2_score(y_val,y_pred_pca))
time4 = time.perf_counter()

# pca.inverse_transform(x_val_pca)

print('\n方差贡献度:',pca.explained_variance_ratio_)
print('方差贡献度占原数据的比例:%.2f%%\n'%(pca.explained_variance_ratio_.sum()*100))

print('无PCA的线性模型计算时间:%.4f毫秒'%(time2*1000-time1*1000))
print('PCA的计算时间:%.4f毫秒'%(time3*1000-time2*1000))
print('有PCA的线性模型计算时间:%.4f毫秒'%(time4*1000-time3*1000))



plt.figure()
plt.plot(y_val,label='val')
# plt.plot(y_pred,label='pred')
plt.plot(y_pred_pca,label='pred_pca')
plt.plot()
plt.legend()
plt.title('boston房价预测,R2=%.3f'%r2_score(y_val,y_pred_pca))
plt.show()