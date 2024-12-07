import pandas as pd
import pickle as pk
import streamlit as st

# Load model and column names
with open(r'D:\DSCP\random_forest_model.pkl', 'rb') as file:
    rf_model = pk.load(file)

with open(r'D:\DSCP\X_train_columns.pkl', 'rb') as file:
    X_train_columns = pk.load(file)

# Load data to retrieve unique values for inputs
cars_data = pd.read_csv(r'D:\DSCP\Cardetails.csv')

# Streamlit UI
st.title('Car Price Prediction')

# Collect inputs
car_name = st.selectbox('Car Brand', cars_data['name'].unique())
year = st.number_input('Year', min_value=1980, max_value=2025, value=2020)
km_driven = st.number_input('Kilometers Driven', min_value=0, value=50000)
fuel = st.selectbox('Fuel Type', ['Diesel', 'Petrol', 'LPG', 'CNG'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.number_input('Mileage (km/l)', min_value=0.0, value=15.0)
engine = st.number_input('Engine (CC)', min_value=0, value=1000)
max_power = st.number_input('Max Power (bhp)', min_value=0, value=90)
seats = st.number_input('Seats', min_value=2, max_value=7, value=5)

# Mapping categorical inputs to numerical values
mapping_dict = {
    'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
    'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
    'transmission': {'Manual': 1, 'Automatic': 2},
    'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5}
}

# Create a DataFrame with user inputs
X_new = pd.DataFrame({
    'name': [car_name],
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [mapping_dict['fuel'][fuel]],
    'seller_type': [mapping_dict['seller_type'][seller_type]],
    'transmission': [mapping_dict['transmission'][transmission]],
    'owner': [mapping_dict['owner'][owner]],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats]
})

# One-hot encode car name and align columns
X_new = pd.get_dummies(X_new, columns=['name'], drop_first=True)
X_new = X_new.reindex(columns=X_train_columns, fill_value=0)

# Predict the price using the trained model
if st.button('Predict Price'):
    price_pred = rf_model.predict(X_new)
    st.write(f"Predicted Selling Price: ₹{price_pred[0]:,.2f}")
