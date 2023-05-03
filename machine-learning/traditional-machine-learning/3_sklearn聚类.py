from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")
#数据加载
iris = load_iris()
data = iris['data']
target = iris['target']

model = KMeans(n_clusters=3)
model.fit(data)
pred = model.predict(data)
print('平均轮廓系数:',silhouette_score(data,pred))



tsne = TSNE(n_components = 3)
data_3d = tsne.fit_transform(data)
fig = plt.figure(figsize=(12, 6), facecolor='w')
ax = fig.add_subplot(121, projection='3d')
ax.scatter(data_3d[target==0][:,0],data_3d[target==0][:,1],data_3d[target==0][:,2],label='class:0')
ax.scatter(data_3d[target==1][:,0],data_3d[target==1][:,1],data_3d[target==1][:,2],label='class:1')
ax.scatter(data_3d[target==2][:,0],data_3d[target==2][:,1],data_3d[target==2][:,2],label='class:2')
plt.title('True')
ax.legend()
ax = fig.add_subplot(122, projection='3d')
ax.scatter(data_3d[pred==0][:,0],data_3d[pred==0][:,1],data_3d[pred==0][:,2],label='class:0')
ax.scatter(data_3d[pred==1][:,0],data_3d[pred==1][:,1],data_3d[pred==1][:,2],label='class:1')
ax.scatter(data_3d[pred==2][:,0],data_3d[pred==2][:,1],data_3d[pred==2][:,2],label='class:2')
plt.title('Pred')
ax.legend()
plt.show()



tsne = TSNE(n_components = 2)
data_2d = tsne.fit_transform(data)
plt.subplot(121)
plt.scatter(data_2d[target==0][:,0],data_2d[target==0][:,1],label='class:0')
plt.scatter(data_2d[target==1][:,0],data_2d[target==1][:,1],label='class:1')
plt.scatter(data_2d[target==2][:,0],data_2d[target==2][:,1],label='class:2')
plt.title('True')
plt.legend()

plt.subplot(122)
plt.scatter(data_2d[pred==0][:,0],data_2d[pred==0][:,1],label='class:0')
plt.scatter(data_2d[pred==1][:,0],data_2d[pred==1][:,1],label='class:1')
plt.scatter(data_2d[pred==2][:,0],data_2d[pred==2][:,1],label='class:2')
plt.title('Pred')
plt.legend()
plt.show()


