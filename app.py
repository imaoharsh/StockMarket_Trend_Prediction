import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data
import datetime as dt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import base64
import hydralit_components as hc
import datetime


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

    st.subheader('Closing Price vs Time Chart with MA200')
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
        
        
            
        
        
        
