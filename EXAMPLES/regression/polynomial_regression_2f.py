
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn; seaborn.set()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
x = np.random.random(100).reshape(50,2);
y = 2 * x[:,0] * x[:,0] -5*x[:,1]  + 1
print(x)
print(y)
fig= plt.figure()
fig.add_subplot(111,projection='3d').scatter(x[:,0],x[:,1], y, 'o');
polynomial_features = PolynomialFeatures(degree=4,include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),("linear_regression", linear_regression)])
pipeline.fit(x,y)
x_test=np.linspace(0, 1, 100).reshape(50,2)
print(x_test)
ax= Axes3D(fig)
print(pipeline.predict(x_test[0]))
ax.scatter(x[:,0],x[:,1],y)
ax.plot(xs=x_test[:,0],ys=x_test[:,1],zs=pipeline.predict(x_test))
plt.show();