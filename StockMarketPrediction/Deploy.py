import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import load_model

import pandas_datareader as data
import streamlit as st

st.title('Stock Trend Prediction')

start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-31", format="%Y-%m-%d"))
end_date = st.date_input("End Date", value=pd.to_datetime("today", format="%Y-%m-%d"))

start = start_date.strftime("%Y/%m/%d")
end = end_date.strftime("%Y/%m/%d")

user_input = st.text_input("Enter Stock Ticker",'ITC.NS')
stock = data.DataReader(user_input,'yahoo',start,end)

st.subheader('Opening Price vs Time chart')
fig = plt.figure(figsize = (10,5))
plt.plot(stock.Open)
st.pyplot(fig)

st.subheader('Opening Price vs Time chart with 100MA')
ma100 = stock.Open.rolling(100).mean()
fig = plt.figure(figsize = (10,5))
plt.plot(ma100,'y')
plt.plot(stock.Open)
st.pyplot(fig)


st.subheader('Opening Price vs Time chart with 100MA & 200MA')
ma100 = stock.Open.rolling(100).mean()
ma200 = stock.Open.rolling(200).mean()
fig = plt.figure(figsize = (10,5))
plt.plot(ma100,'y')
plt.plot(ma200,'r')
plt.plot(stock.Open,'b')
st.pyplot(fig)


stock_training = pd.DataFrame(stock['Open'][0:int(len(stock)*0.80)]) 
stock_testing = pd.DataFrame(stock['Open'][int(len(stock)*80):int(len(stock))]) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
               
model = load_model('test.h5')


past100days = stock_training.tail(100)
final_df = past100days.append(stock_testing , ignore_index =True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)


scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


st.subheader("Predictions vs Original")
plt.figure(figsize = (10,5))
plt.plot(y_test,'r', label = 'Original Price')
plt.plot(y_predicted,'y',label = 'Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)
