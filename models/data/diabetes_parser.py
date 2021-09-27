import numpy as np
import pandas as pd

data = pd.read_csv('diabetes_data_upload.csv')
data = data.replace("Yes", 1)
data = data.replace('No', 0)
data = data.replace('Male', 1)
data = data.replace('Female', 0)
data = data.replace('Positive', 1)
data = data.replace('Negative', 0)

data.to_csv('diabetes.csv', index=False)
