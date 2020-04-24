import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder

def prepare_data(data):
    df = data.copy()
    df['Medical_Tent'] = df['Medical_Tent'].replace(np.nan, 'n/a', regex=True)
    df['Birthday_year'] = df['Birthday_year'].replace(np.nan, -1, regex=True)

    df = df[df['City'].notna()]
    del df['Name']
    df = pd.get_dummies(df, prefix=['Medical_Tent', 'City'])

    del df['Family_Case_ID']

    return df


train_data = pd.read_csv('train.csv', index_col="Patient_ID")
test_data = pd.read_csv('test.csv', index_col="Patient_ID")

train_data = prepare_data(train_data)
test_data = prepare_data(test_data)

train_labels = train_data["Deceased"]
del train_data['Deceased']


x_train, x_val, y_train, y_val = train_test_split(train_data,
                                                  train_labels,
                                                  test_size=0.2,
                                                  random_state=13,
                                                  shuffle=True,
                                                  stratify=train_labels
                                                  )

model = MLPClassifier(solver = 'adam',
                      activation = "logistic",
                      learning_rate = "adaptive",
                      alpha=1e-5,
                      early_stopping = True,
                      max_iter = 200,
                      random_state = 0)


model.fit(x_train, y_train)

labels_val = model.predict(x_val)

print('Results on the validation set:')
print(
    # write your code in here
    classification_report(y_true = y_val, y_pred = labels_val)
)
