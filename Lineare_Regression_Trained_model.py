import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load car dataset
cars_data = pd.read_csv(r'D:\DSCP\Cardetails.csv')

# Extract car brand name (first word as the brand)
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Feature engineering: Convert categorical columns into numeric codes
pd.set_option('future.no_silent_downcasting', True)
cars_data['owner'] = cars_data['owner'].replace(
    ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
    [1, 2, 3, 4, 5]
) 

cars_data['fuel'] = cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4])
cars_data['seller_type'] = cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3])
cars_data['transmission'] = cars_data['transmission'].replace(['Manual', 'Automatic'], [1, 2])

# One-hot encode the 'name' column (car brand) and other categorical features
X = pd.get_dummies(cars_data[['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']])

# Split data into features (X) and target (y)
y = cars_data['selling_price']  # Assuming 'selling_price' is the target column

# Handle missing values by filling them with the median of each column
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Linear Regression): {mse}")

# Calculate R² score
r2_score = lr_model.score(X_test, y_test)
print(f"R² Score (Linear Regression): {r2_score}")

# Save the trained model to a file using pickle
with open(r'D:\DSCP\linear_regression_model.pkl', 'wb') as file:
    pk.dump(lr_model, file)
