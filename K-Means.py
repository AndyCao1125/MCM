from sklearn import cluster
from sklearn import datasets

model = cluster.KMeans(a)
model.fit(X)
y_means = model.predict(X)
plt.scatter(X[:,0],X[:,1], c=y_means, s=50, cmap='rainbow')  #s是超参数
