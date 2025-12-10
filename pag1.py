import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Define the path to your Keras model file
MODEL_PATH = 'D:/Book_cpny/mysql/genre_ann_model.keras'
model = load_model(MODEL_PATH)
model1 = joblib.load('D:/Book_cpny/mysql/tfidf_title.joblib')
model2 = joblib.load('D:/Book_cpny/mysql/label_encoder.joblib')
upld=pd.read_csv('D:/Book_cpny/mysql/demand.csv')




#st.write(upld)
#val=upld['book_id'].count()
#st.write(val)
st.markdown("""
<style>
.big-font {
    font-size: 20px !important;
    font-weight:bold;
}
            
</style>
""", unsafe_allow_html=True)
st.set_page_config(layout="wide")
text1='<p class="big-font">45440</p>'
contain1=st.container(border=True)
contain1.subheader('Total Book sold')
contain1.markdown(text1, unsafe_allow_html=True)
col1, col2, col3, col4,col5,col6 = contain1.columns(6, border=True)
with col1:
    st.write("Fiction")
    st.write(39315)
with col2:
    st.write("History")
    st.write(2078)
with col3:
    st.write("Love")
    st.write(1623)
with col4:
    st.write("Biograph")
    st.write(1460)
with col5:
    st.write("Mystery")
    st.write(547)
with col6:
    st.write("Fantasy")
    st.write(417)
pd_sls_ana=upld.groupby(['book_id'])['price'].sum()
cust_buy=upld.groupby(['customer_id'])['book_id'].count()
#Top 13 book and revenue
top13=pd_sls_ana[pd_sls_ana>=240]
#Top 11 customer buy books
top12=cust_buy[cust_buy>=102]

top1=st.container()
left_col,right_col=top1.columns(2,border=True)
left_col.write('Top 13 book and revenue')
right_col.write('Top 11 customer buy books')
left_col.bar_chart(top13,x_label='Book_id',y_label='Revenue',color="#ff6200")
right_col.bar_chart(top12,x_label='Customer_id',y_label='Book_id',color="#ffaa00")


# You can now use models.Sequential, models.Model, etc.
def predict_genre(title):
    txt = str(title)
    vec = model1.transform([txt]).toarray()
    probs = model.predict(vec)
    idx = probs.argmax(axis=1)[0]
    return model2.inverse_transform([idx])[0]
st.subheader('Movie Title Prediction')
title = st.text_input("Text your book title")
g = predict_genre(title)
if st.button('Predict'):
    st.write(f'The Movie Genre is :',{g})
   
        
    
    


