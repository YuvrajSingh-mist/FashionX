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


model = YOLO('best.pt')

@st.cache_resource
def load_model():
    
    model_vgg16 = VGG16()
    model_vgg16 = Model(inputs=model_vgg16.inputs, outputs=model_vgg16.layers[-2].output)
    # gc.collect()
    return model_vgg16

vgg16 = load_model()
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

filenames = pickle.load(open('images_filenames_myntra_small.pkl', 'rb'))
st.markdown("<h1 style='text-align: center; color: white;'>Recommendations from image</h1>", unsafe_allow_html=True)
st.divider()


features_images = np.array(pickle.load(open('embeddings_myntra_small_latest.pkl', 'rb')))
# features_images = pickle.load(open('features_myntra.pkl', 'rb'))
# final_df = pd.read_csv('myntradataset/styles.csv')
# if os.path.exists('runs/'):
#     os.rmdir('runs/')
# else:
#     pass

# nn = NearestNeighbors(n_neighbors=6,algorithm='brute', metric='euclidean', n_jobs=-1)

# nn.fit(features_images)

# ls = []

def extract(img_path):
#     gc.collect()
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
  
    # get extracted features
    features = vgg16.predict(image, verbose=0).flatten()
    # result = vgg_face.predict(preprocessed_img).flatten()
#     gc.collect()
    return features


def save_uploaded_image(image):
    try:
        with open(os.path.join('uploads', image.name), 'wb') as f:
            f.write(image.getbuffer())
            return True
    except:
        return False

if os.path.exists('uploads/'):
    shutil.rmtree('uploads/')

os.mkdir('uploads/')
        
uploaded_image = st.file_uploader('Upload an image')

ls = []

index_pos_shirts = []
index_pos_pants = []
index_pos_shoes = []
index_pos_shorts = []
index_pos_jacket = []

if uploaded_image is not None:

            
    if save_uploaded_image(uploaded_image):
        if os.path.exists('results'):
            shutil.rmtree('results')
        else:
            os.mkdir('results')
        
        similarity = []
        for file in os.listdir('uploads/'):
            
            features = extract('myntradataset/images/' + file)
            
        for i in range(len(features_images)):
            
            similarity.append((cosine_similarity(features_images[i].reshape(1, -1), features.reshape(1,-1))))
        
        similarity = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i]
        print(similarity)
        
        # for i in range(5):
        #     st.image('myntradataset/images/' + similarity)
#         model.predict(os.path.join('uploads/', uploaded_image.name), save=True,  save_txt=True, save_crop=True, project='results')
        
            
# #         for file in os.listdir('results/predict/crops/'):

# #             if file in ['shirt', 'jacket', 'dress']:

# #                 file2 = 'shirt'

# #                 os.rename('results/predict/crops/{}/'.format(file), 'results/predict/crops/{}/'.format(file2))
        
#         for file in os.listdir('results/predict/crops/'):
#             # indices=[]
#             # ls=[]
#             # print(file)
#             similarity = []
#             for img in os.listdir('results/predict/crops/{}/'.format(file)):
#                 # print(img)
  
#                 # image = Image.open('results/predict/crops/{}/'.format(file)+ img)
#                 # image = preprocess('results/predict/crops/{}/'.format(file)+ img)
#                 # print(img)

#                 # features = resnet50.predict(image).flatten()
#                 features=preprocess('results/predict/crops/{}/'.format(file)+ img)
#                 # print(features)
#                 # distances, indices = nn.kneighbors([features, n_neighbors=5])
#                 # for i in range(len(features_images)):
               
#                 # image = Image.open('results_hed/predict/crops/{}/'.format(file)+ img)
#                 # print(img)

#                 # features = vgg16.predict(image).flatten()
#                 # print(features)
#                 for i in range(len(features_images)):

#                     similarity.append((cosine_similarity(features_images[i].reshape(1, -1), features.reshape(1,-1))))
#                     # print(features)
#                 # print(similarity)
            
#                 # index_pos = 
# #             # index_pos_'{}'.format(file) = []

#                 # indices.append()
#                     # print(features)
#                 # print(similarity)
#                 # for i in range(len(features_images)):

#                     # similarity.append((cosine_similarity(features_images[i].reshape(1, -1), features.reshape(1,-1))))
#                 # print(indices)
#                 # print(distances)
#             # print(indices[0][0])
#                 # print(index_pos)
#             # for i in range(5):
#                 # index_pos_'{}'.format(file) = []
#                 # print(file)
#                 if file == 'shirt':
#                     for i in range(5):
#                         index_pos_shirts.append(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i][0])
#                     # print(distances)

#                 elif file == 'shorts':
#                     for i in range(5):
#                         index_pos_shorts.append(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i][0])
#                     # print(distances)
                
#                 elif file == 'pants':
#                     for i in range(5):
#                         index_pos_pants.append(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i][0])
#                     # print(distances)
                    
                    
#                 elif file == 'shoe':
#                     for i in range(5):
#                         index_pos_shoes.append(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i][0])
#                     # print(distances)
#                 elif file == 'jacket':
#                     for i in range(5):
#                         index_pos_jacket.append(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i][0])
                
#         print(index_pos_pants)
#         print(index_pos_shirts)  
#         print(index_pos_shoes)  
#         print(index_pos_shorts)  
#         # if st.button('Show'):
#         for file in os.listdir('results/predict/crops/'):
#             if file == 'short':
#                 with st.expander('Top 5 recommndations for short'):
#                     if len(index_pos_shorts) != 0:

#                         columns = st.columns(5)
#                         for i in range(len(columns)):
#                             with columns[i]:
#                                 # homepage_url = final_df.iloc[index_pos_shirts[i],2]
#                                 # print(recommendations[i])
#                                 image = cv2.imread('images/{}'.format(filenames[index_pos_shorts[i]]))
#                                 # image = cv2.resize(image, (224, 224))
#                                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                                 # st.write(final_df.iloc[index_pos_short[i],4])

#                                 st.image(image)
#                                 # url = "https://www.streamlit.io"
#                                 # st.write("[Explore](%s)" % homepage_url)

#                     else:
#                         st.write('None')

#             elif file == 'pants':
#                 with st.expander('Top 5 recommndations for Pants'):
#                     if len(index_pos_pants) != 0:

#                         columns = st.columns(5)
#                         for i in range(len(columns)):
#                             with columns[i]:
#                                 # homepage_url = final_df.iloc[index_pos_shirts[i],2]
#                                 # print(recommendations[i])
#                                 image = cv2.imread('images/{}'.format(filenames[index_pos_pants[i]]))
#                                 # image = cv2.resize(image, (224, 224))
#                                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                                 # st.write(final_df.iloc[index_pos_pants[i],4])

#                                 st.image(image)
#                                 # url = "https://www.streamlit.io"
#                                 # st.write("[Explore](%s)" % homepage_url)

#                     else:
#                         st.write('None')

#             elif file == 'shirt':
#                 with st.expander('Top 5 recommndations for Shirts'):
#                     if len(index_pos_shirts) != 0:


#                         columns = st.columns(5)
#                         for i in range(len(columns)):
#                             with columns[i]:
#                                 # homepage_url = final_df.iloc[index_pos_shirts[i],2]
#                                 # print(recommendations[i])
#                                 image = cv2.imread('images/{}'.format(filenames[index_pos_shirts[i]]))
#                                 # image = cv2.resize(image, (224, 224))
#                                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                                 # st.write(final_df.iloc[index_pos_shirts[i],4])

#                                 st.image(image)
#                                 # url = "https://www.streamlit.io"
#                                 # st.write("[Explore](%s)" % homepage_url)


#                     else:
#                         st.write('None')

#             elif file == 'shoe':
#                 with st.expander('Top 5 recommndations for Shoes'):
#                     if len(index_pos_shoes) != 0:

#                         columns = st.columns(5)
#                         for i in range(len(columns)):
#                             with columns[i]:
#                                 # homepage_url = final_df.iloc[index_pos_shirts[i],2]
#                                 # print(recommendations[i])
#                                 image = cv2.imread('images/{}'.format(filenames[index_pos_shoes[i]]))
#                                 # image = cv2.resize(image, (224, 224))
#                                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                                 # st.write(final_df.iloc[index_pos_shoes[i],4])

#                                 st.image(image)
#                                 # url = "https://www.streamlit.io"
#                                 # st.write("[Explore](%s)" % homepage_url)

#                     else:
#                         st.write('None')
#             elif file == 'jacket':
#                 with st.expander('Top 5 recommndations for Jacket'):
#                     if len(index_pos_jacket) != 0:

#                         columns = st.columns(5)
#                         for i in range(len(columns)):
#                             with columns[i]:
#                                 # homepage_url = final_df.iloc[index_pos_shirts[i],2]
#                                 # print(recommendations[i])
#                                 image = cv2.imread('images/{}'.format(filenames[index_pos_jacket[i]]))
#                                 # image = cv2.resize(image, (224, 224))
#                                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                                 # st.write(final_df.iloc[index_pos_shoes[i],4])

#                                 st.image(image)
#                                 # url = "https://www.streamlit.io"
#                                 # st.write("[Explore](%s)" % homepage_url)

#                     else:
#                         st.write('None')

