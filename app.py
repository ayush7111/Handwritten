#--> importing required libraries
import streamlit as st
import numpy as np
import pandas as pd
from pandas import read_csv
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

from keras.models import load_model

# label letter map
import string
upper_case = string.ascii_uppercase
label_letter_map = {}
for idx, letter in enumerate(upper_case):
  label_letter_map[idx] = letter

#--> for Logo and Title
col1, mid, col2 = st.columns([1,15,25])
with col1:
    st.image('data/logo.jpeg', width=150)
with col2:
    st.title('Handwritten Character Recognition')

#--> creating a block which can expand to give more info
with st.expander("About The Project"):
     st.write("""
         This Project (web app) is designed to recognize the alphabets from images of handwritten alphabets.
         Algorithm Used - CNN
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
         The dataset used for this project is A-Z Handwritten Dataset from NIST taken from kaggle.
         * It Has-
            - 372450 Rows
            - 785 Columns 
        * 80% Data is used for Training and 20 % Data is used of Testing purpose
     """)
#loading the saved model
model = load_model("./data/text_model.h5")

st.subheader('')
st.subheader('')

#--> User Input
st.subheader('Upload The Image Of Alphabet-: ')
uploaded_file = st.file_uploader("Choose a file")
img_array=[]
if uploaded_file is not None:
     #st.write(uploaded_file)
     with open(os.path.join("./data/test", uploaded_file.name), "wb") as f:
         f.write(uploaded_file.getbuffer())
     image = Image.open(uploaded_file)
     st.image(image, caption='Input Image')
     img_path = os.path.join("./data/test", uploaded_file.name)
     img_array =cv2.imread(img_path)


alpha_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
ret, gray = cv2.threshold(gray, 75, 180, cv2.THRESH_BINARY)
element = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 90))
gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, element)
gray = gray / 255.  # downsampling
gray = cv2.resize(gray, (28, 28))  # resizing
# reshaping the image
gray = np.reshape(gray, (28, 28))

#--> Displayong Result
pred = alpha_dict[np.argmax(model.predict(np.reshape(gray, (1, 28, 28, 1))))]
st.subheader('')
st.text('Recognised As - > ')
st.write(pred)


