import pandas as pd
import numpy as np

class Dataset():
    def __init__(self):
        self.train_data = pd.read_csv('train.csv', index_col="Patient_ID")
        self.test_data = pd.read_csv('test.csv', index_col="Patient_ID")



        self.train_data = Dataset.prepare_data(self.train_data)
        self.test_data = Dataset.prepare_data(self.test_data)

        self.train_data = Dataset.create_features(self.train_data)
        self.test_data = Dataset.create_features(self.test_data)

        self.target = self.train_data['Deceased']
        self.test_data['Medical_Tent_T'] = 0
        del self.train_data['Deceased']




    def create_features(data):
        df_encoded = data.copy()
        df_encoded["family_size"] = [df_encoded.Family_Case_ID.value_counts()[fid] for fid in df_encoded.Family_Case_ID.values]

        return df_encoded

    def prepare_data(data):
        df = data.copy()
        gender = {'Mr.':"M",
                  'Ms.':"F",
                  'Master':"M",
                  'Miss':"F",
                  "Mrs.":"F"}

        df['Name'] = df['Name'].str.rsplit(' ', n=0, expand=True)
        df['Gender'] = [gender[item] for item in df['Name']]

        del df['Name']

        df['Medical_Tent'] = df['Medical_Tent'].replace(np.nan, 'n/a', regex=True)
        df['Birthday_year'] = df['Birthday_year'].replace(np.nan, -1, regex=True)

        df = df[df['City'].notna()]
        df = pd.get_dummies(df, prefix=['Medical_Tent', 'City', 'Gender'])
        del df['Gender_F']

        return df
