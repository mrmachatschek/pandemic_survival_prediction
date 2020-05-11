import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import tree
from dataset import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import KFold

class ModelSVM():
    def __init__(self, vector_model = svm.NuSVC):
        dataset = Dataset()

        print(f"Model SVM - {vector_model}")
        self.model = vector_model()

        self.train_data = dataset.train_data
        self.test_data = dataset.test_data
        self.target = dataset.target

        self.selected_variables = [
            'Severity',
            'Gender_M',
            'City_Albuquerque',
            'City_Santa Fe',
            "severity_against_avg_gender",
            'Medical_Tent_n/a',
            "Sev_family",
            'spending_family_member',
            'family_size',

            # 'Birthday_year',
            # 'Parents or siblings infected',
            # 'Wife/Husband or children infected',
            # 'Medical_Tent_A',
            # 'Medical_Tent_B',
            # 'Medical_Tent_C',
            # 'Medical_Tent_D',
            # 'Medical_Tent_E',
            # 'Medical_Tent_F',
            # 'Medical_Tent_G',
            # 'Medical_Tent_T',
            # 'Medical_Expenses_Family',
            # 'City_Taos',
            # 'spending_vs_severity',
            # "severity_against_avg_city",
            # "severity_against_avg_tent",
            # "spending_family_severity",


            ]

        scaler = RobustScaler().fit(self.test_data[self.selected_variables])

        self.train_data[self.selected_variables] = scaler.transform(
                               self.train_data[self.selected_variables])

        self.test_data[self.selected_variables] = scaler.transform(
                                self.test_data[self.selected_variables])

        self.test_data = self.test_data[self.selected_variables]
        self.train_data = self.train_data[self.selected_variables]

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
            self.test_data["Deceased"]

# model = ModelSVM(vector_model = svm.NuSVC)
# model.run_model(path="solution.csv")
