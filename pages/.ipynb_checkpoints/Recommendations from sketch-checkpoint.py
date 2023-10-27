import streamlit as st
import time
import pandas  as pd
import pickle
import nltk
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card
import tensorflow as tf
import keras
import pandas
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import expand_dims


st.set_page_config(
    page_title = 'FashionX'
)

st.markdown("<h1 style='text-align: center; color: white;'>Recommendations from sketch</h1>", unsafe_allow_html=True)
st.divider()


hed_features_ls = pickle.load(open('extracted_features_hed_images.pkl', 'rb'))

if os.path.exists('uploads2/'):
    os.rimdir('uploads2/')
else:
    os.mkdir('uploads2/')
#####detect from web cam the sketch 


##resize the image
def preprocess(img):
    image = cv2.imread(img)
    image = cv2.resize(image,(224, 224))
    exp_image = expand_dims(image, axis=0)
    preprocessed_image = preprocess_input(exp_image)
    return preprocessed_image

#detection of edges from sketch
ls = []


model = YOLO('best.pt')
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224,224, 3), pooling='avg')
model.predict(os.path.join('uploads2/', uploaded_image), save=True, save_txt=True, save_crop=True)





for img in os.listdir('hed_images/'):
    try:
                image = Image.open(img)
                image = preprocess(image)


                features = vgg16.predict(image).flatten()

                for feature in range(len(features_images)):

                    similarity = ls.append((cosine_similarity(hed_features_ls[i][0].reshape(1, -1), features.rehsape(1,-1)), hed_features_ls[i][1]))

                for i in range(5):
                    index_pos + '_{}'.format(i).append(sorted(similarity, reverse=True, key=lambda x:x[1])[i][0])
            except:
                pass
#cosine similarity