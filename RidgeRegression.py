import pandas as pd
from sklearn.metrics import root_mean_squared_error
from ucimlrepo import fetch_ucirepo
from sklearn import linear_model
from Preprocessing import Preprocessor

adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

data = pd.concat([X, y], axis = 1)

preprocessor = Preprocessor()
X_trn, X_tst, y_trn, y_tst = preprocessor.preprocess(data)

reg = linear_model.Ridge(alpha=.1)

reg.fit(X_trn, y_trn)

prediction = reg.predict(X_tst)

print(f"Accuracy: {reg.score(X_tst, y_tst):.2f}")
print(f"RMSE: {root_mean_squared_error(y_tst, prediction)}")
