import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_model():
    with open('weather_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

st.title("Weather Forecasting App")

st.header("Input Weather Features")

temperature = st.number_input("Temperature (°C)", step=0.1)
dew_point_temp = st.number_input("Dew Point Temperature (°C)", step=0.1)
humidity = st.number_input("Relative Humidity (%)", step=0.1)
wind_speed = st.number_input("Wind Speed (km/h)", step=0.1)
visibility = st.number_input("Visibility (km)", step=0.1)
pressure = st.number_input("Pressure (kPa)", step=0.1)

if st.button("Forecast Weather"):
    model = load_model()
    input_data = np.array([[temperature, dew_point_temp, humidity, wind_speed, visibility, pressure]])
    prediction = model.predict(input_data)
    st.subheader("Forecasted Weather Condition:")
    if prediction[0] == 0:
        st.write('Weather is "CLEAR"')
    elif prediction[0] == 1:
        st.write('Weather is "CLOUDY"')
    elif prediction[0] == 2:
        st.write('Weather is "RAIN"')
    else:
        st.write('Weather is "SNOWY"')

if st.checkbox("Save a new model (for developers)"):
    st.write("You can save your trained model using the following code in Jupyter Notebook:")
    st.code('''import pickle

with open('weather_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully as 'weather_model.pkl'")''', language='python')

if st.checkbox("How to load the model in Python"):
    st.write("You can load your saved model using the following code:")
    st.code('''with open('weather_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

forecast = loaded_model.predict([[25, 65, 10, 1013]])
print("Forecasted weather:", forecast)''', language='python')

st.write("Upload your trained weather model as 'weather_model.pkl' in the same directory.")

