import streamlit as st
import pickle
import numpy as np
from PIL import Image


def load_model():
    with open ('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
le_airline = data['airline']
le_source_city = data['source_city']
le_departure_time = data['departure_time']
le_stops = data['stops']
le_arrival_time = data['arrival_time']
le_destination_city = data['destination_city']
le_class = data['Class']


def prediction_page():
    st.image(image=Image.open('ec_utb.png'))
    st.image(image=Image.open('plane.png'), caption= 'Price prediction')
    st.title('Agile Assignment 2023 / Prediction of flight prices :seat::heavy_dollar_sign::airplane_departure:')
    st.header('  Made by Jesper Hedlund / Mousumi Deb / Ilias Outras ')
    st.write('Please fill bellow the details from the flight that you want to predict the ticket price')
    airline = st.selectbox('Select Airline', ('Vistara', 'Air_India', 'Indigo', 'Go_First', 'AirAsia'))
    source_city = st.selectbox('Select departure city', ('Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad'))
    departure_time =st.selectbox('Select departure time', ('Morning', 'Early_Morning', 'Evening', 'Night', 'Afternoon'))
    stops =st.selectbox('Select number of stops', ('one', 'zero', 'two_or_more'))
    arrival_time =st.selectbox('Select arrival time', ('Night', 'Evening', 'Morning', 'Afternoon', 'Early_Morning'))
    destination_city = st.selectbox('Select destination city', ('Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad'))
    Class =st.selectbox('Select economy class', ('Economy', 'Business'))
    days_left = st.slider('Select days left for the flight', 0, 365)
    duration = st.slider('Select duration of the flight (h)', 0, 50)
    ok = st.button('Predict price')
    if ok:
        X = np.array([[airline, source_city,departure_time, stops, arrival_time, destination_city, Class, days_left, duration]])
        X[:, 0] = le_airline.transform(X[:, 0])
        X[:, 1] = le_source_city.transform(X[:, 1])
        X[:, 2] = le_departure_time.transform(X[:, 2])
        X[:, 3] = le_stops.transform(X[:, 3])
        X[:, 4] = le_arrival_time.transform(X[:, 4])
        X[:, 5] = le_destination_city.transform(X[:, 5])
        X[:, 6] = le_class.transform(X[:, 6])
        X= X.astype(float)

        price = regressor.predict(X)
        st.subheader(f'The estmated price is{price[0]: .2f}')

prediction_page()


