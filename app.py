import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data
import datetime as dt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


start='2010-01-01'
end='2019-12-31'



st.title('Stock Market Trend Prediction And Analysis')

user_input=st.text_input('Enter Stock Ticker','AAPL')
df = data.DataReader(user_input, 'stooq')


#Describing Data

st.subheader('Date from 2010 - 2024')

st.write(df.describe())


st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with MA100')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with MA100')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


#Splitting data into Training and Testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.50)])                           #70%
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.50):int(len(df))])                 #30%
print(data_training.shape)
print(data_testing.shape)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))


data_training_array=scaler.fit_transform(data_training)

    
model=load_model('stock_model.keras')


past_100_days=data_training.tail(100)

final_df=past_100_days.append(data_testing,ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)

scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test *scale_factor



st.subheader('Predicted Price vs Original Price')
fig2=plt.figure(figsize=(12, 6))  # Use the 'figsize' parameter to set the figure size

plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')

# Additional plot settings or annotations can be added here

plt.legend()  # Add legend if necessary
st.pyplot(fig2)