# Create some simple data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

np.random.seed(0)
X = np.random.random(20)
print(X)
y = 3 * X.squeeze() + 2 + np.random.randn(20)
print(y)
plt.plot(X.squeeze(), y, 'o');
#plt.show();
#============LinearRegression=======
#model = LinearRegression();
#model.fit(X,y);

#X_fit = np.linspace(0,1,10)[:,np.newaxis]
#y_fit = model.predict(X_fit)

#plt.plot(X.squeeze(),y,'o')
#plt.plot(X_fit.squeeze(),y_fit);
#============PolynomialRegression===
polynomial_features = PolynomialFeatures(degree=6)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
pipeline.fit(X[:, np.newaxis], y)
X_test = np.linspace(0, 1, 100)
plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")

plt.show();
