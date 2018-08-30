from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.decomposition import PCA

iris = load_iris()
X, y = iris.data, iris.target
k_means = KMeans(n_clusters=3, random_state=0) # Fixing the RNG in kmeans
k_means.fit(X)
y_pred = k_means.predict(X)

pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)

pl.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred,cmap='RdYlBu');
pl.show()