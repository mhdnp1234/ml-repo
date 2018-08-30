from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

digits = datasets.load_digits()
digits.images.shape
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary')
    ax.text(0.05, 0.05, str(digits.target[i]),transform=ax.transAxes, color='green')
    ax.set_xticks([])
    ax.set_yticks([])
print(digits.images.shape)
print(digits.data.shape)
print(digits.target)
iso = Isomap(n_components=2)
data_projected = iso.fit_transform(digits.data)
data_projected.shape
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10));
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)

Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,random_state=2)
print(Xtrain.shape, Xtest.shape)
clf = LogisticRegression(penalty='l2')
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
accuracy_score(ytest, ypred)
print(confusion_matrix(ytest, ypred))
#plt.imshow(np.log(confusion_matrix(ytest, ypred)),cmap='Blues', interpolation='nearest')
#plt.grid(False)
#plt.ylabel('true')
#plt.xlabel('predicted');

fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8, 8), cmap='binary')
    ax.text(0.05, 0.05, str(ypred[i]),transform=ax.transAxes,color='green' if (ytest[i] == ypred[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()