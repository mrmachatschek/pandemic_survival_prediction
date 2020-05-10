import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import tree
from dataset import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold

class ModelSVM():
    def __init__(self):
        self.dataset = Dataset()
        self.model = svm.SVC(C=1.0,
                             kernel='rbf',
                             degree=3,
                             gamma='scale',
                             coef0=0.0,
                             shrinking=True,
                             probability=False,
                             tol=0.001,
                             cache_size=200,
                             class_weight=None,
                             verbose=False,
                             max_iter=-1,
                             decision_function_shape='ovr',
                             break_ties=False,
                             random_state=0)
                             
        k_fold = KFold(n_splits=5)

        self.y_train = []
        self.x_train = []

        self.y_test = []
        self.x_test = []

        for train_index, test_index in k_fold.split(self.dataset.train_data):
            X_train, X_test = self.dataset.train_data.iloc[train_index], self.dataset.train_data.iloc[test_index]
            Y_train, Y_test = self.dataset.target.iloc[train_index], self.dataset.target.iloc[test_index]
            self.x_train.append(X_train)
            self.x_test.append(X_test)
            self.y_train.append(Y_train)
            self.y_test.append(Y_test)

    def train_model(self):
        for index in range(5):
            self.model.fit(self.x_train[index],self.y_train[index])
            print(self.model.score(self.x_test[index], self.y_test[index]))


    def predict_results(self,path="solution.csv"):
        labels_val = self.model.predict(self.dataset.test_data)
        self.dataset.test_data.insert(loc=0, column='Deceased', value=labels_val)
        self.dataset.test_data["Deceased"].to_csv(path)

model = ModelSVM()
model.train_model()
model.predict_results()
