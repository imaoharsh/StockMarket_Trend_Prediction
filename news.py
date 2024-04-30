import pycountry
import streamlit as st
import requests as req



st.title("BuzzNation News Hub")


col1,col2=st.columns([3,1])
with col1:
    user=st.text_input('Enter country name','India')
with col2:
    cat=st.radio('choose news category',('Tech','politics','Sports','Business'))
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
        st.image(article['urlToImage'])
        
    
    
    
