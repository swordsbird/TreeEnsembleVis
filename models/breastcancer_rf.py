# breast cancer RF model
# Test 0.9790209790209791
# Train 1.0

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

random_state = 114
data_table = pd.read_csv('./data/cancer.csv')

label = ['B', 'M']
labels = data_table['diagnosis']
labels = labels.apply(lambda x: 1 if x == 'M' else 0)
data_table = data_table.drop(['diagnosis', 'id'], axis=1)
X = data_table.values
y = labels.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state, stratify=labels)


# sm = SMOTE(random_state=random_state)
# X_train, y_train = sm.fit_resample(X_train, y_train)

parameters = {
        'n_estimators': 200,
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
