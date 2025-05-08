import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LearningCurveDisplay
from sklearn.naive_bayes import GaussianNB
from ucimlrepo import fetch_ucirepo
from Preprocessing import Preprocessor
from sklearn.metrics import ConfusionMatrixDisplay as CMD


adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

data = pd.concat([X, y], axis = 1)

preprocessor = Preprocessor()
X_trn, X_tst, y_trn, y_tst = preprocessor.preprocess(data)
print(y_tst.unique())

bayes = GaussianNB()
bayes.fit(X_trn, y_trn)

y_hat = bayes.predict(X_tst)
print(f"Accuracy: {accuracy_score(y_tst, y_hat)}")


LearningCurveDisplay.from_estimator(estimator=bayes, X = X_trn, y = y_trn)

disp = CMD(confusion_matrix=confusion_matrix(y_tst, y_hat),
          display_labels=["<=50K", ">50K"])
disp.plot()
plt.show()