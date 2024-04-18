import numpy as np
from sklearn.model_selection import train_test_split


class Stacking:
    def __init__(self, base_models, meta):
        self.base_models = base_models
        self.meta = meta

    def fit(self,X,y):
        X_train, X_vald, y_train, y_vald = train_test_split(X,y, test_size=.4, random_state=42)
        base_prediction = []
        for model in self.base_models:
            model.fit(X_train,y_train)
            base_prediction.append(model.predict(X_vald))

        base_prediction = np.column_stack(base_prediction)
        self.meta.fit(base_prediction, y_vald)

    def predict(self,X):
        base_prediction = np.column_stack([model.predict(X) for model in self.base_models])
        return self.meta.predict(base_prediction)