# fetch dataset
import pandas as pd
from ucimlrepo import fetch_ucirepo

import Preprocessing
from Preprocessing import Preprocessor

adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

data = pd.concat([X, y], axis = 1)

preprocesor = Preprocessor()
data = preprocesor.preprocess(data)

