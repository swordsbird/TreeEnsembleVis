# wine RF model
# Test 0.915
# Train 0.9966052376333656
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

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
    'n_estimators': 300,
    'max_depth': 13,
    'random_state': random_state,
    'max_features': 'auto',
    'oob_score': True,
}

clf = RandomForestClassifier(**parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Test')
print('Accuracy Score is', accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_train)

print('Train')
print('Accuracy Score is', accuracy_score(y_train, y_pred))

