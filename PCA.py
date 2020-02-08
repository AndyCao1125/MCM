from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

pca = PCA(n_components=)   #人为确定主要贡献方向
pca.fit(X)

#图像化PCA
plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.5)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=3)
plt.axis('equal');
print('explained_variance:\n\t', pca.explained_variance_ratio_)

#如果是为了保留百分之多少的误差
model = PCA(0.95)
model.fit(X)
print (model.transform(X))