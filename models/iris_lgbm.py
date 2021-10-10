# Iris LightGBM model
# Test 0.9736842105263158
# Train 0.9821428571428571
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from lightgbm import LGBMClassifier
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder

random_state = 44
pd.set_option("max_columns",100)
pd.set_option("max_rows",900)
pd.set_option("max_colwidth",200)
df = pd.read_csv('data/Iris.csv')
df = df.drop_duplicates()
df.drop(columns="Id", inplace=True)

X = df.drop(columns="Species")
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)

parameters = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'num_leaves': 100,
    'max_depth': 7,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.02,
    'bagging_fraction': 0.8,
    'bagging_freq': 30,
    'feature_fraction': 0.5,
}

clf = LGBMClassifier(**parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Test')
print('Accuracy Score is', accuracy_score(y_test, y_pred))

y_pred_train = clf.predict(X_train)

print('Train')
print('Accuracy Score is', accuracy_score(y_train, y_pred_train))
