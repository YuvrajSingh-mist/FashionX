import streamlit as st
import os
import time
import pandas  as pd
import pickle
import nltk
import spacy
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card
from PIL import Image
import tensorflow as tf
import keras
import cv2
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow import expand_dims
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import joblib
from tensorflow.keras.layers import GlobalMaxPooling2D
from keras import Sequential
from keras.layers import Dense, Flatten
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image

from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import gc




# @st.cache_resource
def vgg():
    vgg16 = VGG16(weights='imagenet', input_shape = (224,224,3), include_top=False)
    for layers in vgg16.layers:
        layers.trainable=False
    
    return vgg16

# gc.collect()

# resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224, 3), pooling='avg')
# resnet50.trainable = False
# print(resnet50.summary())
# resnet50 = Sequential()
# resnet50.add(conv_base)
# resnet50.add(GlobalMaxPooling2D())

# st.set_page_config(
#     page_title = 'FashionX'
# )
st.markdown("<h1 style='text-align: center; color: white;'>Recommendations from image</h1>", unsafe_allow_html=True)
st.divider()

# @st.cache_data
def features_image():
    features_images = np.array(pickle.load(open('embeddings_images_15000_recommend_vgg16.pkl', 'rb')))
    return features_images
# features_images = pickle.load(open('features_myntra.pkl', 'rb'))
# final_df = pd.read_csv('myntradataset/styles.csv')
# if os.path.exists('runs/'):
#     os.rmdir('runs/')
# else:
#     pass

# nn = NearestNeighbors(n_neighbors=6,algorithm='brute', metric='euclidean', n_jobs=-1)

# nn.fit(features_images)

# ls = []
# @st.cache_resource
model = YOLO('best.pt')
    
   

       
    
def extract(img_path, vgg16):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    
    result = vgg16.predict(preprocessed_img, verbose=0).flatten()
    return result


def save_uploaded_image(image):
    try:
        with open(os.path.join('uploads', image.name), 'wb') as f:
            f.write(image.getbuffer())
            return True
    except:
        return False

    