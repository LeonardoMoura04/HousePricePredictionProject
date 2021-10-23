# Step 1: Import libraries, functions, classes, etc...
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import joblib

# Step 2: Import data
train = pd.read_csv('Files/House_Price_Prediction/train.csv')
x_test = pd.read_csv('Files/House_Price_Prediction/test.csv')
y_test = pd.read_csv('Files/House_Price_Prediction/sample_submission.csv')

# Step 3: Clearing Data
x_train = train.drop(['POSTED_BY', 'BHK_OR_RK', 'ADDRESS', 'LATITUDE', 'LONGITUDE', 'TARGET(PRICE_IN_LACS)'], axis=1)
y_train = train['TARGET(PRICE_IN_LACS)']
x_test = x_test.drop(['POSTED_BY', 'BHK_OR_RK', 'ADDRESS', 'LATITUDE', 'LONGITUDE'], axis=1)

# Step 4: Instantiate Model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Step 5: Train model
predictions = lr.predict(x_test)

# Save your model
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")