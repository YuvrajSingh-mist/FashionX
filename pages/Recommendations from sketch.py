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
from numpy.linalg import norm
from PIL import Image
import tensorflow as tf
import keras
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import expand_dims
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from tensorflow.keras.layers import GlobalMaxPooling2D
from keras import Sequential
from keras.layers import Dense, Flatten
from numpy.linalg import norm



from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import gc

# model = YOLO('best.pt')
# conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224, 3))
# conv_base.trainable = False

# resnet50 = Sequential()
# resnet50.add(conv_base)
# resnet50.add(GlobalMaxPooling2D())
def vgg():
    vgg16 = VGG16(weights='imagenet', input_shape = (224,224,3), include_top=False)
    for layers in vgg16.layers:
        layers.trainable=False
    
    return vgg16

st.set_page_config(
    page_title = 'FashionX'
)

st.markdown("<h1 style='text-align: center; color: white;'>Recommendations from sketches</h1>", unsafe_allow_html=True)
st.divider()

def features_image_drawings():
    features_images = pickle.load(open('embeddings_images_20000_recommend_drawings_vgg16.pkl', 'rb'))
    return features_images

def features_image_sketch():
    features_images = pickle.load(open('sketches_8000_embeddings.pkl', 'rb'))
    return features_images


final_df = pd.read_csv('final_df.csv')
# if os.path.exists('runs/'):
#     os.rmdir('runs/')
# else:
#     pass

filenames_drawings = pickle.load(open('images_recommend_20000_filenames_drawings.pkl', 'rb'))
filenames_sketch = pickle.load(open('sketches_8000_filenames.pkl', 'rb'))

def extract(img_path, vgg16):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    
    result = vgg16.predict(preprocessed_img, verbose=0).flatten()
    return result


def save_uploaded_image_outline(image):
    try:
        with open(os.path.join('uploads2/', image.name), 'wb') as f:
            f.write(image.getbuffer())
            return True
    except:
        return False
    
def save_uploaded_image_sketch(image):
    try:
        with open(os.path.join('uploads3/', image.name), 'wb') as f:
            f.write(image.getbuffer())
            return True
    except:
        return False
    


        
ls = []

index_pos_shirts = []
index_pos_pants = []
index_pos_shoes = []
index_pos_shorts = []

index_pos = []

if st.checkbox('Upload a drawing/outline'):
    if os.path.exists('uploads2/'):
        shutil.rmtree('uploads2/')
        os.mkdir('uploads2/')
    # else:
    else:
        os.mkdir('uploads2/')
        
    uploaded_image = st.file_uploader('Upload an outline/drawing')

    if uploaded_image is not None:

        if save_uploaded_image_outline(uploaded_image):
        # if save_uploaded_image(uploaded_image):
        #     if os.path.exists('results_sketches'):
        #         shutil.rmtree('results_sketches')
            # else:
            #     os.mkdir('results_sketches')

            # model.predict(os.path.join('uploads/', uploaded_image.name), save=True, save_txt=True, save_crop=True, project='results_sketches')


    #         for file in os.listdir('results_hed/predict/crops/'):

    #             if file in ['shirt', 'jacket', 'dress']:

    #                 file2 = 'shirt'

    #                 os.rename('results_hed/predict/crops/{}/'.format(file), 'results_hed/predict/crops/{}/'.format(file2))

    #         for file in os.listdir('uploads2/'):

    #            # print(file)
            similarity=[]
            features_images_drawings = features_image_drawings()
            # if st.checkbox('Recommend'):
            #     for file in os.listdir('results/predict/crops/'):
            similarity = []
            if st.checkbox('Recommend'):
                for img in os.listdir('uploads2/'):

                    vgg16 = vgg()
                    # gc.collect()
                    features = extract('uploads2/' + img, vgg16)

                    for i in range(len(features_images_drawings)):

                        similarity.append((cosine_similarity(features.reshape(1,-1), features_images_drawings[i].reshape(1, -1))))

                    similarity = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
                    print(similarity)

                    if st.checkbox('Show'):
                        for file in os.listdir('uploads2/'):
    #                         if file == 'short':
                                with st.expander('Top 10 recommndations'):
    #                                 if len(index_pos_shorts) != 0:

                                    columns = st.columns(10)
                                    for i in range(len(columns)):
                                        with columns[i]:
                                            temp = ' '.join(filenames_drawings[similarity[i][0]].split('/')[-1:])
                                            temp = temp.split('.')[-2]
                                            print(temp)
                                            # print(filenames[similarity[0][0]].split('/')[-1:])
                                            # path='/'.join(filenames_drawings[similarity[i][0]].split('/')[-1:])
                                            # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                            print('images/' + temp + '.jpg')
                                            image = cv2.imread(('images/' + temp + '.jpg'))
                                            # image = cv2.resize(image, (224, 224))
                                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                            # st.write(final_df.iloc[index_pos_shirts[i],4])

                                            st.image(image)

                    # else:
                    #         st.write('None')

st.divider()


# else:


if st.checkbox('Upload a sketch'):
    
    if os.path.exists('uploads3/'):
        shutil.rmtree('uploads3/')
        os.mkdir('uploads3/')
    # else:
    else:
        os.mkdir('uploads3/')
        
    uploaded_image = st.file_uploader('Upload a sketch')

    if uploaded_image is not None:
        if save_uploaded_image_sketch(uploaded_image):

        # if save_uploaded_image(uploaded_image):
        #     if os.path.exists('results_sketches'):
        #         shutil.rmtree('results_sketches')
            # else:
            #     os.mkdir('results_sketches')

            # model.predict(os.path.join('uploads/', uploaded_image.name), save=True, save_txt=True, save_crop=True, project='results_sketches')


    #         for file in os.listdir('results_hed/predict/crops/'):

    #             if file in ['shirt', 'jacket', 'dress']:

    #                 file2 = 'shirt'

    #                 os.rename('results_hed/predict/crops/{}/'.format(file), 'results_hed/predict/crops/{}/'.format(file2))

    #         for file in os.listdir('uploads2/'):

    #            # print(file)
            similarity=[]
            features_images_sketch = features_image_sketch()
        
            if st.checkbox('Recommend'):
                # for file in os.listdir('results/predict/crops/'):
                #     similarity = []
                for img in os.listdir('uploads3/'):

                    vgg16 = vgg()
                    # gc.collect()
                    features = extract('uploads3/' + img, vgg16)

                    for i in range(len(features_images_sketch)):

                        similarity.append((cosine_similarity(features.reshape(1,-1), features_images_sketch[i].reshape(1, -1))))

                    similarity = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
                    # print(similarity)

                    if st.checkbox('Show'):
                        # for file in os.listdir('uploads/'):
                        #     if file == 'short':
                        with st.expander('Top 10 recommndations'):
                    
                            columns = st.columns(10)
                            for i in range(len(columns)):
                                with columns[i]:
                                    # print(filenames_sketch[similarity[i][0]].split('/')[-1:])
                                    temp = ' '.join(filenames_sketch[similarity[i][0]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    # print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    # path='/'.join(filenames_sketch[similarity[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])

                                    st.image(image)

                            # else:
                            #     st.write('None')

st.divider()