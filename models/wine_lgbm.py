# wine LightGBM model
# Test 0.9225
# Train 0.9995150339476236
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

random_state = 132
data_table = pd.read_csv('./data/winequality-red.csv')

labels = data_table['quality']
labels = labels.apply(lambda x: 1 if x > 6.5 else 0)
y = labels.values
data_table = data_table.drop(['quality'], axis=1)
X = data_table.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state)

sm = SMOTE(random_state=random_state)
X_train, y_train = sm.fit_resample(X_train, y_train)

parameters = {
    'n_estimators': 400,
    'learning_rate': 0.25,
    'num_leaves': 200,
    'max_depth': 10,
    'min_data_in_leaf': 200,
    'lambda_l1': 0.1,
    'lambda_l2': 10,
}

clf = LGBMClassifier(**parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Test')
print('Accuracy Score is', accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_train)

print('Train')
print('Accuracy Score is', accuracy_score(y_train, y_pred))

