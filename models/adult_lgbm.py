# adult LightGBM model
# Test 0.8557220527781462
# Train 0.9222804024239571
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

random_state = 114
adults = pd.read_csv('data/adult.csv')

adults = adults.replace(to_replace="?", value=np.nan)
adults = adults.dropna()
adults = adults.reset_index(drop=True)

adults.rename(columns={"education.num": "education_num", "marital.status": "marital_status",
                       "capital.gain": "capital_gain", "capital.loss": "capital_loss",
                       "hours.per.week": "hours_per_week", "native.country":"native_country"},
              inplace=True)

adults.drop(columns=["education"], inplace=True)

numerical_columns = adults.select_dtypes(include=[np.number]).columns
categorical_columns = adults.select_dtypes(exclude=[np.number]).columns
categorical_columns = categorical_columns.difference(["income"])

scaler = StandardScaler()
encoder = OneHotEncoder(sparse=False)
std_num_adults = pd.DataFrame(scaler.fit_transform(adults[numerical_columns]), columns=numerical_columns)
ohe_cat_adults = pd.DataFrame(encoder.fit_transform(adults[categorical_columns]))

scaled_encoded_adults = pd.concat([std_num_adults, ohe_cat_adults], axis=1)

X = scaled_encoded_adults.values
y = adults["income"].apply(lambda income: 1 if income == ">50K" else 0).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state)

sm = SMOTE(random_state=random_state)
X_train, y_train = sm.fit_resample(X_train, y_train)

parameters = {
        'n_estimators': 400,
        'learning_rate': 0.2,
        'num_leaves': 250,
        'max_depth': 10,
        'min_data_in_leaf': 400,
        'lambda_l1': 0.65,
        'lambda_l2': 1,
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
