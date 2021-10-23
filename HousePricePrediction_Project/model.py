
# Step 1: Import libraries, functions, classes, etc...
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib

# Step 2: Import data
df = pd.read_csv('Files/Housing/Housing.csv')

X = df.drop(['price'], axis=1)
y = df['price']

le = preprocessing.LabelEncoder()
le.fit(X['furnishingstatus'])
X['furnishingstatus'] = le.transform(X['furnishingstatus'])

# Step 3: Clearing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Step 4: Instantiate Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 5: Train model
predictions = lr.predict(X_test)

# Save your model
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")