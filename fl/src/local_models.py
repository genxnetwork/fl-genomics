import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_train = np.load('/home/dkolobok/Downloads/X_train.npy')
y_train = np.load('/home/dkolobok/Downloads/y_train.npy')

model = LinearRegression()
# model = ElasticNet()
# model = Lasso()

# model = SVR() # works
# model = BayesianRidge() # no
# model = SGDRegressor() # works
# model = KernelRidge() # no

# model = XGBRegressor(max_depth=2)

print("Training")
model.fit(X_train, y_train)
# model.fit(X_train, y_train, eval_set=(X_val, y_val))

# y_preds_test = model.predict(X_test)
y_preds_train = model.predict(X_train)
print(f"train r2: {r2_score(y_train, y_preds_train)}")
# print(f"test r2: {r2_score(y_test, y_preds_test)}")
pass