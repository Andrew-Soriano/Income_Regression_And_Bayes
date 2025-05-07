import pandas as pd
from ucimlrepo import fetch_ucirepo
from seaborn import pairplot as pairplot
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder


# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

adult_df = pd.concat([X, y], axis = 1)
#print(adult_df)

# metadata
#print(adult.metadata)

# variable information
#print(adult.variables)

#pairplot(adult_df, hue='income')
#plt.show()

enc = OrdinalEncoder()
enc.set_output(transform="pandas")
enc.fit(X)
print(enc.categories)
X_scaled = enc.transform(X)
print(X)
print(X_scaled)
scaler = preprocessing.StandardScaler()
scaler.set_output(transform="pandas")
scaler.fit(X_scaled)
X_scaled = scaler.transform(X_scaled)
adult_df_scaled = pd.concat([X_scaled, y], axis = 1)

#pairplot(adult_df_scaled, hue='income')
#plt.show()
