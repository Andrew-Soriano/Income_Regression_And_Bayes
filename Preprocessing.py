import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


class Preprocessor:
    def preprocess(self, data) -> pd.DataFrame:
        #Drop all missing data
        data = data.dropna()

        #Fix the duplicate classes
        data = data.replace(to_replace=['<=50K.', ">50K."], value=['<=50K', ">50K"])

        #Map classes to integers
        enc = OrdinalEncoder()
        enc.set_output(transform="pandas")
        enc.fit(data)
        data = enc.transform(data)

        return train_test_split(data.drop('income', axis = 1), data['income'])