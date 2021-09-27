# bank LightGBM model
# Test 0.9750733137829912
# Train 1.0
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

random_state = 114
data_table = pd.read_csv('data/bank.csv')

X = data_table.drop('Bankrupt?', axis=1).values
y = data_table['Bankrupt?'].values
features = data_table.drop('Bankrupt?', axis=1).columns.to_list()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=random_state)

sm = SMOTE(random_state=random_state)
X_train, y_train = sm.fit_resample(X_train, y_train)

parameters = {
        'n_estimators': 400,
        'learning_rate': 0.25,
        'num_leaves': 250,
        'max_depth': 11,
        'min_data_in_leaf': 320,
        'lambda_l1': 0.65,
        'lambda_l2': 1.08,
        'bagging_fraction': 0.71,
        'bagging_freq': 8,
        'feature_fraction': 0.27,
}

clf = LGBMClassifier(**parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Test')
print('Accuracy Score is', accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_train)

print('Train')
print('Accuracy Score is', accuracy_score(y_train, y_pred))
