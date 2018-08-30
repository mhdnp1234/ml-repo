
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn; seaborn.set()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
model = LinearRegression(normalize=True)
print(model.normalize)
print(model)
x = np.random.random(200).reshape(100,2);
y = 2 * x[:,0] -5*x[:,1]  + 1
print(x)
print(y)
fig= plt.figure()
fig.add_subplot(111,projection='3d').scatter(x[:,0],x[:,1], y, 'o');
model.fit(x, y)
x_test=np.linspace(0, 1, 200).reshape(100,2)
print(x_test)
ax= Axes3D(fig)
print(model.predict(x_test[0]))
ax.scatter(x[:,0],x[:,1],y)
ax.plot(xs=x_test[:,0],ys=x_test[:,1],zs=model.predict(x_test))
#plt.figure().add_subplot(111,projection='3d').plot(x_test[:,np.newaxis],model.predict(x_test))
print(model.coef_)
print(model.intercept_)
print(model)
plt.show();