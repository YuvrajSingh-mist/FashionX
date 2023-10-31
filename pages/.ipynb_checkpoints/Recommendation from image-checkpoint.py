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
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import expand_dims
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

model = YOLO('best.pt')
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224,224, 3), pooling='avg')


st.set_page_config(
    page_title = 'FashionX'
)

st.markdown("<h1 style='text-align: center; color: white;'>Recommendations from image</h1>", unsafe_allow_html=True)
st.divider()


features_images = pickle.load(open('features_all.pkl', 'rb'))
final_df = pd.read_csv('final_df.csv')
# if os.path.exists('runs/'):
#     os.rmdir('runs/')
# else:
#     pass


def preprocess(img):
    image = cv2.imread(img)
    image = cv2.resize(image,(224, 224))
    exp_image = expand_dims(image, axis=0)
    preprocessed_image = preprocess_input(exp_image)
    return preprocessed_image



def save_uploaded_image(image):
    try:
        with open(os.path.join('uploads', image.name), 'wb') as f:
            f.write(image.getbuffer())
            return True
    except:
        return False
    
uploaded_image = st.file_uploader('Upload an image')

ls = []

index_pos_shirts = []
index_pos_pants = []
index_pos_shoes = []
index_pos_shorts = []


if uploaded_image is not None:

            
    if save_uploaded_image(uploaded_image):
        if os.path.exists('results'):
            shutil.rmtree('results')
        else:
            os.mkdir('results')
            
        model.predict(os.path.join('uploads/', uploaded_image.name), save=True, save_txt=True, save_crop=True, project='results')
        
            
        for file in os.listdir('results/predict/crops/'):

            if file in ['shirt', 'jacket', 'dress']:

                file2 = 'shirt'

                os.rename('results/predict/crops/{}/'.format(file), 'results/predict/crops/{}/'.format(file2))
        
        for file in os.listdir('results/predict/crops/'):
            similarity=[]
            ls=[]
            # print(file)
            for img in os.listdir('results/predict/crops/{}/'.format(file)):
                # print(img)
  
                image = Image.open('results/predict/crops/{}/'.format(file)+ img)
                image = preprocess('results/predict/crops/{}/'.format(file)+ img)
                # print(img)

                features = vgg16.predict(image).flatten()
                # print(features)
                for i in range(len(features_images)):

                    similarity.append((cosine_similarity(features_images[i][0].reshape(1, -1), features.reshape(1,-1)), int(features_images[i][1])))
                    # print(features)
                # print(similarity)
            for i in range(10):
                # index_pos_'{}'.format(file) = []
                # print(file)
                if file == 'shirt':
                    index_pos_shirts.append(sorted(enumerate(similarity), reverse=True, key=lambda x:x[1])[i][1][1])
                
                elif file == 'shorts':
                    index_pos_shorts.append(sorted(enumerate(similarity), reverse=True, key=lambda x:x[1])[i][1][1])
                
                elif file == 'pants':
                    index_pos_pants.append(sorted(enumerate(similarity), reverse=True, key=lambda x:x[1])[i][1][1])
                
                elif file == 'shoe':
                    index_pos_shoes.append(sorted(enumerate(similarity), reverse=True, key=lambda x:x[1])[i][1][1])

                
        print(index_pos_pants)
        print(index_pos_shirts)  
        print(index_pos_shoes)  
        print(index_pos_shorts)  
        if st.button('Show'):
            for file in os.listdir('results/predict/crops/'):
                if file == 'short':
                    with st.expander('Top 5 recommndations for short'):
                        if len(index_pos_short) != 0:
 
                            columns = st.columns(10)
                            for i in range(len(columns)):
                                with columns[i]:
                                    homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                    # print(recommendations[i])
                                    image = cv2.imread('images_recommend/{}'.format(index_pos_short[i]) + '.png')
                                    image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    st.write(final_df.iloc[index_pos_short[i],4])

                                    st.image(image)
                                    # url = "https://www.streamlit.io"
                                    st.write("[Explore](%s)" % homepage_url)
                                    
                        else:
                            st.write('None')
                
                elif file == 'pants':
                    with st.expander('Top 5 recommndations for Pants'):
                        if len(index_pos_pants) != 0:
                            
                            columns = st.columns(10)
                            for i in range(len(columns)):
                                with columns[i]:
                                    homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                    # print(recommendations[i])
                                    image = cv2.imread('images_recommend/{}'.format(index_pos_pants[i]) + '.png')
                                    image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    st.write(final_df.iloc[index_pos_pants[i],4])

                                    st.image(image)
                                    # url = "https://www.streamlit.io"
                                    st.write("[Explore](%s)" % homepage_url)
                        
                        else:
                            st.write('None')

                elif file == 'shirt':
                    with st.expander('Top 5 recommndations for Shirts'):
                        if len(index_pos_shirts) != 0:

                            
                            columns = st.columns(10)
                            for i in range(len(columns)):
                                with columns[i]:
                                    homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                    # print(recommendations[i])
                                    image = cv2.imread('images_recommend/{}'.format(index_pos_shirts[i]) + '.png')
                                    image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    st.write(final_df.iloc[index_pos_shirts[i],4])

                                    st.image(image)
                                    # url = "https://www.streamlit.io"
                                    st.write("[Explore](%s)" % homepage_url)
                    

                        else:
                            st.write('None')
                
                elif file == 'shoe':
                    with st.expander('Top 5 recommndations for Shoes'):
                        if len(index_pos_shoes) != 0:

                            columns = st.columns(10)
                            for i in range(len(columns)):
                                with columns[i]:
                                    homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                    # print(recommendations[i])
                                    image = cv2.imread('images_recommend/{}'.format(index_pos_shoes[i]) + '.png')
                                    image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    st.write(final_df.iloc[index_pos_shoes[i],4])

                                    st.image(image)
                                    # url = "https://www.streamlit.io"
                                    st.write("[Explore](%s)" % homepage_url)
                                
                        else:
                            st.write('None')

