#--> importing required libraries

import os
import cv2
import string
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import streamlit as st
from pandas import read_csv
from keras.models import load_model

#--> Setting Page Layout
st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

#--> label letter map
upper_case = string.ascii_uppercase
label_letter_map = {}
for idx, letter in enumerate(upper_case):
  label_letter_map[idx] = letter

#--> for Logo and Title
#col1, mid, col2 = st.columns([1,15,25])
col1, mid, col2 = st.columns([1,8,25])
with col1:
    st.image('data/logo.jpeg', width=150)
with col2:
    st.title('Detection of Hindi Language Handwritten Characters')

#--> creating a block which can expand to give more info
with st.expander("About The Project"):
     st.write("""
         This Project (web app) is designed to recognize the alphabets from images of handwritten alphabets.
         Algorithm Used - Logistic Regression
         * This Project is Scalable ( as generated model can be reused )
         * Modular in approach
         * Code is Easy to read and Understand
     """)

#--> for proper spacing between elements
st.subheader('')
st.subheader('')

st.subheader('Dataset-: ')
#--> including explation of the dataset using expander
with st.expander("See explanation"):
     st.write("""
         The dataset used for this project is Devanagari Handwritten Dataset. 
        * 80% Data is used for Training and 20 % Data is used of Testing purpose
     """)

#loading the saved model
#model = pickle.load(open("./data/model.pkl",rb))

st.subheader('')
st.subheader('')

#--> User Input
st.subheader('Upload the image-: ')
uploaded_file = st.file_uploader("Choose a file")
img_array=[]
if uploaded_file is not None:
     #st.write(uploaded_file)
     with open(os.path.join("./data/test", uploaded_file.name), "wb") as f:
         f.write(uploaded_file.getbuffer())
     image = Image.open(uploaded_file)
     st.image(image, caption='Input Image', width=200)
     img_path = os.path.join("./data/test", uploaded_file.name)
     img_array =cv2.imread(img_path)
     decode={0:'ka' ,1:'kha' ,2:'ga',3:'gha',4:'kna',5:'cha' }
     gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
     gray = cv2.medianBlur(gray, 5)
     ret, gray = cv2.threshold(gray, 75, 180, cv2.THRESH_BINARY)
     element = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 90))
     gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, element)
     gray = gray / 255.  # downsampling
     gray = cv2.resize(gray, (32, 32))  # resizing
     # reshaping the image
     gray = np.reshape(gray, (32, 32))

    #--> Displayong Result
     pred = decode[np.argmax(model.predict(np.reshape(gray, (1, 28, 28, 1))))]
     st.subheader('')
     st.markdown('<p class="big-font">Predicted as  - ></p>', unsafe_allow_html=True)
     st.write(pred)
