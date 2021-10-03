# diabetes RF model
# Test 0.9769230769230769
# Train 1.0
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

random_state = 114
data_table = pd.read_csv('data/diabetes.csv')

X = data_table.drop('class', axis=1).values
y = data_table['class'].values
features = data_table.drop('class', axis=1).columns.to_list()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state)
sm = SMOTE(random_state=random_state)
X_train, y_train = sm.fit_resample(X_train, y_train)

parameters = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': random_state,
    'max_features': None,
}

clf = RandomForestClassifier(**parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Test')
print('Accuracy Score is', accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_train)

print('Train')
print('Accuracy Score is', accuracy_score(y_train, y_pred))
