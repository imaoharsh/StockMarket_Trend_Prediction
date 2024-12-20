import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data
import datetime as dt
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import base64
import hydralit_components as hc

from keras.models import load_model

import nasdaqdatalink as link

link.read_key(filename='key.txt')
from streamlit_navigation_bar import st_navbar

page = st_navbar(["Home", "News", "Insights", "Community", "About"])
#st.write(page)


if page == "Home":
    
    main_bg="back.jpeg"
    main_bg_ext="png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-repeat: no-repeat;
            background-size: cover;
            background-color: 'E4DEBE';
            
            
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


        
    



    st.title(':white[Stock Market Trend Prediction And Analysis]')
    
    

    user_input=st.text_input('Enter Stock Ticker','NSE/TATASTEEL')
    #df = data.DataReader(user_input, 'stooq')

    df = link.get(user_input)
    
   
    
    
    
    
    
    #Describing Data

    

    st.write(df.describe())


    st.subheader('Closing Price vs Time Chart')
    fig=plt.figure(figsize=(12,8))
    plt.plot(df.Close)
    st.pyplot(fig)


    st.subheader('Closing Price vs Time Chart with MA100')
    ma100=df.Close.rolling(100).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart ')
    ma100=df.Close.rolling(100).mean()
    ma200=df.Close.rolling(200).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100,label='MA100')
    plt.plot(ma200,label='Predicted Price')
    plt.plot(df.Close)
    
    plt.legend()
    st.pyplot(fig)


    #Splitting data into Training and Testing
    data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.50)])                           #70%
    data_testing=pd.DataFrame(df['Close'][int(len(df)*0.50):int(len(df))])                 #30%
    print(data_training.shape)
    print(data_testing.shape)


    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))


    data_training_array=scaler.fit_transform(data_training)

    #ML    
    

  


elif page == "Insights":
    main_bg="back.jpeg"
    main_bg_ext="png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-repeat: no-repeat;
            background-size: cover;
            background-color: 'E4DEBE';
            
            
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("This is a page for title")

elif page=="News":
    main_bg="back.jpeg"
    main_bg_ext="png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-repeat: no-repeat;
            background-size: cover;
            background-color: 'E4DEBE';
            
            
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
   
    import pycountry
    import requests as req



    st.title("BuzzNation News Hub")


    col1,col2=st.columns([3,1])
    with col1:
        user=st.text_input('Enter country name','India')
    
        
    btn=st.button('Enter')
    apiKEY="a81e1e4d25a74bcca5d568645cb66ecb"

    if btn:
        country=pycountry.countries.get(name=user).alpha_2
        
        url=f"https://newsapi.org/v2/top-headlines?country={country}&category=business&apiKey={apiKEY}"
        
        r=req.get(url)
        r=r.json()
        articles=r['articles']
        for article in articles:
            st.header(article['title'])
            st.write(article['publishedAt'])
            if article['author']:
                st.write(article['author'])
            st.write(article['source']['name'])
            st.write(article['description'])
            st.write(article['url'])
    
elif page == "Community":
    main_bg="back.jpeg"
    main_bg_ext="png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-repeat: no-repeat;
            background-size: cover;
            background-color: 'E4DEBE';
            
            
        }}
        </style>
        """,
        unsafe_allow_html=True,
        )
    st.title('This is a community page')


elif page == "About":
    main_bg="back.jpeg"
    main_bg_ext="png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-repeat: no-repeat;
            background-size: cover;
            background-color: 'E4DEBE';
            
            
        }}
        </style>
        """,
        unsafe_allow_html=True,
        )
    st.title('About')
    st.write('This is a minor project created by Harshit, Harsh Goyal,Satwik Shukla')
        
        
            
        
        
        
