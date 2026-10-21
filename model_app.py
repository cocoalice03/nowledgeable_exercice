import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load("regression.joblib")

st.title("House Price Prediction")

# Create input fields
size = st.number_input("Size (m²)", min_value=0.0, value=100.0)
nb_rooms = st.number_input("Number of rooms", min_value=0, value=2, step=1)
garden = st.number_input("Garden (0 or 1)", min_value=0, max_value=1, value=0, step=1)

# Make prediction when button is clicked
if st.button("Predict Price"):
    # Create input DataFrame
    input_data = pd.DataFrame([[size, nb_rooms, garden]], columns=['size', 'nb_rooms', 'garden'])
    
    # Predict
    prediction = model.predict(input_data)
    
    # Display result
    st.write(f"Predicted price: {prediction[0]:,.2f} €")
