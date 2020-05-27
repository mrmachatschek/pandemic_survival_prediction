import pandas as pd
import numpy as np

'''
'Family_Case_ID',
'Severity',
'Birthday_year',
'Parents or siblings infected',
'Wife/Husband or children infected',
'Medical_Expenses_Family',
'Medical_Tent_A',
'Medical_Tent_B',
'Medical_Tent_C',
'Medical_Tent_D',
'Medical_Tent_E',
'Medical_Tent_F',
'Medical_Tent_G',
'Medical_Tent_T',
'Medical_Tent_n/a',
'City_Albuquerque',
'City_Santa Fe',
'City_Taos',
'Gender_M',
'family_size'
'''

# Severity Against Age
# Medical Expenses Against Severity

# Sibling Infected / Family Size
# Severity Against Severity in Medical Tent
# Severity Against Severity in City

class Dataset():
    def __init__(self):
        self.train_data = pd.read_csv('data/train.csv', index_col="Patient_ID")
        self.test_data = pd.read_csv('data/test.csv', index_col="Patient_ID")

    def apply_preparation(self):
        self.train_data = Dataset.prepare_data(self.train_data)
        self.test_data = Dataset.prepare_data(self.test_data)

        self.train_data = Dataset.create_features(self.train_data)
        self.test_data = Dataset.create_features(self.test_data)

        self.target = self.train_data['Deceased']
        self.test_data['Medical_Tent_T'] = 0
        del self.train_data['Deceased']

    def create_gender(self):
        gender = {'Mr.':"M",
                  'Ms.':"F",
                  'Master':"M",
                  'Miss':"F",
                  "Mrs.":"F"}


        self.train_data['Name'] = self.train_data['Name'].str.rsplit(' ', n=0, expand=True)
        self.train_data['Gender'] = [gender[item] for item in self.train_data['Name']]

        self.test_data['Name'] = self.test_data['Name'].str.rsplit(' ', n=0, expand=True)
        self.test_data['Gender'] = [gender[item] for item in self.test_data['Name']]

    def create_features(data):
        df_encoded = data.copy()
        df_encoded["family_size"] = [df_encoded.Family_Case_ID.value_counts()[fid] for fid in df_encoded.Family_Case_ID.values]
        df_encoded["spending_vs_severity"] = df_encoded["Medical_Expenses_Family"] / df_encoded["Severity"]
        df_encoded["spending_family_member"] = df_encoded["Medical_Expenses_Family"] / (df_encoded['Parents or siblings infected'] + df_encoded['Wife/Husband or children infected'] + 1)
        df_encoded["severity_against_avg_city"] = df_encoded["Severity"] / df_encoded["Sev_by_city"]
        df_encoded["severity_against_avg_tent"] = df_encoded["Severity"] / df_encoded["Sev_by_tent"]
        df_encoded["severity_against_avg_gender"] = df_encoded["Severity"] / df_encoded["Sev_by_gender"]
        df_encoded["spending_family_severity"] = ( df_encoded["Medical_Expenses_Family"] / (df_encoded['Parents or siblings infected'] + df_encoded['Wife/Husband or children infected'] + 1) ) / df_encoded["Severity"]

        return df_encoded

    def prepare_data(data):
        df = data.copy()
        gender = {'Mr.':"M",
                  'Ms.':"F",
                  'Master':"M",
                  'Miss':"F",
                  "Mrs.":"F"}
        df['Medical_Tent'] = df['Medical_Tent'].replace(np.nan, 'n/a', regex=True)
        df['Birthday_year'] = df['Birthday_year'].replace(np.nan, -1, regex=True)

        df['Name'] = df['Name'].str.rsplit(' ', n=0, expand=True)
        df['Gender'] = [gender[item] for item in df['Name']]

        del df['Name']

        mean = df[["Severity",'City']].groupby(['City']).mean()
        mean["Sev_by_city"] = mean["Severity"]
        del mean["Severity"]
        df = df.join(mean["Sev_by_city"], on='City', how='left')

        mean = df[["Severity",'Medical_Tent']].groupby(['Medical_Tent']).mean()
        mean["Sev_by_tent"] = mean["Severity"]
        del mean["Severity"]
        df = df.join(mean["Sev_by_tent"], on='Medical_Tent', how='left')

        mean = df[["Severity",'Gender']].groupby(['Gender']).mean()
        mean["Sev_by_gender"] = mean["Severity"]
        del mean["Severity"]
        df = df.join(mean["Sev_by_gender"], on='Gender', how='left')

        mean = df[["Severity",'Family_Case_ID']].groupby(['Family_Case_ID']).mean()
        mean["Sev_family"] = mean["Severity"]
        del mean["Severity"]
        df = df.join(mean["Sev_family"], on='Family_Case_ID', how='left')

        df = df[df['City'].notna()]
        df = pd.get_dummies(df, prefix=['Medical_Tent', 'City', 'Gender'])
        del df['Gender_F']

        return df
