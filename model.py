import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import tree
from dataset import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import KFold

class Model():
    def __init__(self,
                 model = None,
                 variables = None,
                 train_set = None,
                 test_set = None,
                 target = None
                 ):

        if(model==None):
            self.model = svm.NuSVC()
        else:
            self.model = model

        print(f"Model - {self.model}")

        if (train_set is None):
            dataset = Dataset()
            self.train_data = dataset.train_data
            self.test_data = dataset.test_data
            self.target = dataset.target

            if(variables == None):
                self.selected_variables = self.train_data.columns
            else:
                self.selected_variables = variables

            scaler = RobustScaler().fit(self.test_data[self.selected_variables])

            self.train_data[self.selected_variables] = scaler.transform(
                                   self.train_data[self.selected_variables])

            self.test_data[self.selected_variables] = scaler.transform(
                                    self.test_data[self.selected_variables])

            self.test_data = self.test_data[self.selected_variables]
            self.train_data = self.train_data[self.selected_variables]

        else:
            if(variables == None):
                self.selected_variables = train_set.columns
            else:
                self.selected_variables = variables

            self.train_data = train_set[self.selected_variables]
            self.test_data = test_set[self.selected_variables]
            self.target = target



    def run_model(self,path = None):
        self.create_kfolds()
        self.train_model()
        self.predict_results(path)

    def draw_boxplots(self):
        train_data = self.train_data.join(self.target,how='left')
        boxplot_1 = train_data.boxplot(column=[
                                               'Severity',
                                               'Birthday_year',
                                               'Parents or siblings infected',
                                               'Wife/Husband or children infected',
                                               ],
                                       by=["Deceased"])

        boxplot_2 = train_data.boxplot(column=[
                                                'Medical_Expenses_Family',
                                                'City_Albuquerque',
                                                'City_Santa Fe',
                                                'City_Taos',
                                               ],
                                       by=["Deceased"])

        boxplot_3 = train_data.boxplot(column=[
                                                'Gender_M',
                                                'family_size',
                                                'spending_vs_severity',
                                                'spending_family_member',
                                              ],
                                       by=["Deceased"])

        boxplot_4 = train_data.boxplot(column=[
                                                "severity_against_avg_city",
                                                "severity_against_avg_tent",
                                                "Sev_family",
                                                "spending_family_severity",
                                              ],
                                       by=["Deceased"])
        boxplot_5 = train_data.boxplot(column=['Medical_Tent_G','Medical_Tent_T','Medical_Tent_n/a',"severity_against_avg_gender"],by=["Deceased"])

        boxplot_6 = train_data.boxplot(column=[
                                               'Medical_Tent_A',
                                               'Medical_Tent_B',
                                               'Medical_Tent_C',
                                               'Medical_Tent_D',
                                               'Medical_Tent_E',
                                               'Medical_Tent_F',
                                             ],
                                      by=["Deceased"])
        plt.show()


    def create_kfolds(self):
        k_fold = KFold(n_splits=5)
        self.y_train = []
        self.x_train = []
        self.y_test = []
        self.x_test = []

        for train_index, test_index in k_fold.split(self.train_data):
            X_train = self.train_data.iloc[train_index]
            X_test = self.train_data.iloc[test_index]
            Y_train = self.target.iloc[train_index]
            Y_test = self.target.iloc[test_index]
            self.x_train.append(X_train)
            self.x_test.append(X_test)
            self.y_train.append(Y_train)
            self.y_test.append(Y_test)

    def train_model(self):
        average_accuracy = 0
        best_accuracy = 0
        best_model = None
        for index in range(5):
            self.model.fit(self.x_train[index],self.y_train[index])
            score = self.model.score(self.x_test[index], self.y_test[index])
            if(score > best_accuracy):
                best_accuracy = score
                best_model = index

            average_accuracy = average_accuracy + score

        average_accuracy = average_accuracy / 5

        print("Average model accuracy: {:2.2%}".format(average_accuracy))
        print("Highest model accuracy: {:2.2%}".format(best_accuracy))
        self.model.fit(self.x_train[best_model],self.y_train[best_model])


    def predict_results(self,path = None):
        labels_val = self.model.predict(self.test_data)
        self.test_data.insert(loc=0, column='Deceased', value=labels_val)
        if (path != None):
            self.test_data["Deceased"].to_csv(path)
            print(f"Solution set saved as '{path}'.")
        else:
            print(f"Solution not saved.")
