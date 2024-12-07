import pandas as pd
import pickle as pk
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load car dataset
cars_data = pd.read_csv(r'D:\DSCP\Cardetails.csv')

# Convert categorical columns to numeric codes
mapping_dict = {
    'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
    'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
    'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
    'transmission': {'Manual': 1, 'Automatic': 2}
}
cars_data.replace(mapping_dict, inplace=True)

# One-hot encode car name and other categorical columns
X = pd.get_dummies(cars_data[['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']])
y = cars_data['selling_price']

# Save column names to ensure consistent feature alignment during prediction
X_train_columns = X.columns

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2_score = rf_model.score(X_test, y_test)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2_score}")

# Save model and columns
with open(r'D:\DSCP\random_forest_model.pkl', 'wb') as file:
    pk.dump(rf_model, file)

with open(r'D:\DSCP\X_train_columns.pkl', 'wb') as file:
    pk.dump(X_train_columns, file)
