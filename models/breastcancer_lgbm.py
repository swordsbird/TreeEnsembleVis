# breast cancer LightGBM model
# Test 0.993006993006993
# Train 1.0
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

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
    'n_estimators': 100,
    'learning_rate': 0.1,
    'num_leaves': 20,
    'max_depth': 5,
}

clf = LGBMClassifier(**parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Test')
print('Accuracy Score is', accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_train)

print('Train')
print('Accuracy Score is', accuracy_score(y_train, y_pred))
