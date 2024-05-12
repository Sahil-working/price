import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf


st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock Id" , "GOOG")

from datetime import datetime

end = datetime.now()
start = datetime(end.year - 20 , end.month , end.day)

google_data = yf.download(stock , start ,end)

splitting_len = int(len(google_data)*0.7)

x_test = pd.DataFrame(google_data.Close[splitting_len : ])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100 , len(scaled_data)):
    x_data.append(scaled_data[i-100 : i])
    y_data.append(scaled_data[i])

x_data , y_data = np.array(x_data) , np.array(y_data)

x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128 , return_sequences = True , input_shape = (x_train.shape[1],1)))
model.add(LSTM(64 , return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer= 'adam' , loss = 'mean_squared_error')

model.fit(x_train , y_train)


st.subheader("Stock data")
st.write(google_data)


def plot_graph(figsize ,values , column_name,extra = 0 , extra_dataset = 0) :
    fig = plt.figure(figsize = figsize)
    plt.plot(values , 'Orange')
    plt.plot(column_name.Close , 'b')
    if extra:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6) ,google_data['MA_for_250_days'] , google_data,0))

st.subheader('Original Close price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6) ,google_data['MA_for_200_days'] , google_data,0))

st.subheader('Original Close price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6) ,google_data['MA_for_100_days'] , google_data,0))

st.subheader('Original Close price and MA for 250 days and MA for 100 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6) ,google_data['MA_for_250_days'] , google_data,1,google_data['MA_for_100_days']))



predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test  = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame(
    {
        'Original_data ' : inv_y_test.reshape(-1),
        'predicted' : inv_pre.reshape(-1)
    }
    , index = google_data.index[splitting_len+100 : ]

)

st.subheader("Original Values vs Predicted Values")
st.write(plotting_data)

st.subheader("Original Close price vs Predicted Close Price")
fig = plt.figure(figsize = (15,6))
plt.plot(pd.concat([google_data.Close[ : splitting_len+100] ,plotting_data] , axis = 0))
plt.legend(["Data - Not used " , "Original Test Data" , "Predicted Stock Data"])
st.pyplot(fig)
