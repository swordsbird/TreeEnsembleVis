# abalone LightGBM model
# Test 0.90311004784689
# Train 0.9332535169111045
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

data = pd.read_csv("data/abalone.csv", sep=",", header='infer')

category = np.repeat("empty000", data.shape[0])
for i in range(0, data["Rings"].size):
    if data["Rings"][i] <= 7:
        category[i] = "G1"
    elif data["Rings"][i] > 7:
        category[i] = "G2"

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['Sex'])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

data = data.drop(['Sex'], axis=1)
data['category_size'] = category
data = data.drop(['Rings'], axis=1)

features = data.iloc[:,np.r_[0:7]]
labels = data.iloc[:,7]

X_train, X_test, y_train, y_test, X_gender, X_gender_test = \
    train_test_split(features, labels, onehot_encoded, random_state=10, test_size=0.2)

temp = X_train.values
X_train_gender = np.concatenate((temp, X_gender), axis=1)

temp = X_test.values
X_test_gender = np.concatenate((temp, X_gender_test), axis=1)

parameters = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'num_leaves': 25,
    'max_depth': 5,
    'min_data_in_leaf': 200,
    # 'lambda_l1': 0.65,
    # 'lambda_l2': 2,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'feature_fraction': 0.9,
}

clf = LGBMClassifier(**parameters)
clf.fit(X_train_gender, y_train)

y_pred = clf.predict(X_test_gender)

print('Test')
print('Accuracy Score is', accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_train_gender)

print('Train')
print('Accuracy Score is', accuracy_score(y_train, y_pred))
