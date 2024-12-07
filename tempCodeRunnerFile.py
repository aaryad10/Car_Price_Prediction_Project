import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib

# Set Matplotlib to use TkAgg backend for GUI-based interactive plotting (only if running as a standalone script)
matplotlib.use('TkAgg')  # Uncomment this line if running as a standalone script

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

# Initialize the Support Vector Regression model
svr_model = SVR(kernel='rbf')  # Using the radial basis function (RBF) kernel

# Train the model
svr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svr_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (SVR): {mse}")

# Calculate R² score
r2_score = svr_model.score(X_test, y_test)
print(f"R² Score (SVR): {r2_score}")

# Save the trained model to a file using pickle
with open(r'D:\DSCP\svr_model.pkl', 'wb') as file:
    pk.dump(svr_model, file)

# Visualization 1: Residual Plot
residuals = y_test - y_pred

# Plotting residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue', alpha=0.6)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for SVR Model')
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# Visualization 2: Feature Importance-like plot based on correlation with target variable
correlation_matrix = X.join(y).corr()

# Plot top 10 correlated features with 'selling_price'
top_corr_features = correlation_matrix['selling_price'].sort_values(ascending=False).head(10)

# Plotting feature importance-like bar plot
plt.figure(figsize=(10, 6))
top_corr_features.drop('selling_price').plot(kind='barh', color='skyblue')
plt.xlabel('Correlation with Selling Price')
plt.title('Top 10 Features Correlated with Selling Price')
plt.gca().invert_yaxis()  # Invert to have the highest correlation on top
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)
