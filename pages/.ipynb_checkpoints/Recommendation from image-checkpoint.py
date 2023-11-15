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
def model():
    model = YOLO('best.pt')
    
    model.predict(os.path.join('uploads/', uploaded_image.name), save=True,  save_txt=True, save_crop=True, project='results')
    
    
    
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

if os.path.exists('uploads/'):
    shutil.rmtree('uploads/')

else:
    os.mkdir('uploads/')
        


ls = []

index_pos_shirts = []
index_pos_pants = []
index_pos_shoes = []
index_pos_shorts = []
index_pos_jacket = []
filenames = pickle.load(open('images_recommend_15000_filenames.pkl', 'rb'))
if st.checkbox('Upload a single photo: '):
    uploaded_image = st.file_uploader('Upload an image')
    if uploaded_image is not None:
        
        if save_uploaded_image(uploaded_image):
          

            if os.path.exists('results'):
                shutil.rmtree('results')
            
            model()
                
            # gc.collect()

#                 for file in os.listdir('results/'):

#                     features = extract('results/' + file)

               
                # print(similarity)

                # for i in range(5):
                #     st.image('myntradataset/images/' + similarity)



#                 for file in os.listdir('results/predict/crops/'):

#                     if file in ['shirt', 'jacket', 'dress']:

#                         file2 = 'shirt'

#                         os.rename('results/predict/crops/{}/'.format(file), 'results/predict/crops/{}/'.format(file2))
        features_images = features_image()
        if st.checkbox('Recommend'):
            for file in os.listdir('results/predict/crops/'):
                similarity = []
                for img in os.listdir('results/predict/crops/{}/'.format(file)):
                   
                    vgg16 = vgg()
                    # gc.collect()
                    features = extract('results/predict/crops/{}/'.format(file) + img, vgg16)
                    
                    for i in range(len(features_images)):

                        similarity.append((cosine_similarity(features.reshape(1,-1), features_images[i].reshape(1, -1))))

                    similarity = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
                    
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
                    # for i in range(5):
                        # index_pos_'{}'.format(file) = []
                        # print(file)
                if file == 'shirt':
                    for i in range(10):
                        print(similarity[i][0])
                        index_pos_shirts.append(similarity[i][0])
                    # print(distances)

                elif file == 'shorts':
                    for i in range(10):
                        index_pos_shorts.append(similarity[i][0])
                    # print(distances)

                elif file == 'pants':
                    for i in range(10):
                        index_pos_pants.append(similarity[i][0])
                    # print(distances)


                elif file == 'shoe':
                    for i in range(10):
                        index_pos_shoes.append(similarity[i][0])
                    # print(distances)
                elif file == 'jacket':
                    for i in range(10):
                        index_pos_jacket.append(similarity[i][0])

            print(index_pos_pants)
            print(index_pos_shirts)  
            print(index_pos_shoes)  
            print(index_pos_shorts)  

            # if st.checkbox('Show'):
            for file in os.listdir('results/predict/crops/'):
                if file == 'short':
                    with st.expander('Top 5 recommndations for short'):
                        if len(index_pos_shorts) != 0:

                            columns = st.columns(10)
                            for i in range(len(columns)):
                                with columns[i]:
                                    temp = ' '.join(filenames[index_pos_shorts[i]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    path='/'.join(filenames[similarity[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])

                                    st.image(image)

                        else:
                            st.write('None')

                elif file == 'pants':
                    with st.expander('Top 5 recommndations for Pants'):
                        if len(index_pos_pants) != 0:

                            columns = st.columns(10)
                            for i in range(len(columns)):
                                with columns[i]:
                                    temp = ' '.join(filenames[index_pos_pants[i]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    path='/'.join(filenames[similarity[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])

                                    st.image(image)

                        else:
                            st.write('None')

                elif file == 'shirt':
                    with st.expander('Top 5 recommndations for Shirts'):
                        if len(index_pos_shirts) != 0:


                            columns = st.columns(10)
                            for i in range(len(columns)):
                                with columns[i]:
                                    # homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                      # print(filenames[similarity[i][0]])
                                    temp = ' '.join(filenames[index_pos_shirts[i]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    path='/'.join(filenames[similarity[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])

                                    st.image(image)
#                                     # url = "https://www.streamlit.io"
#                                     # st.write("[Explore](%s)" % homepage_url)


                        else:
                            st.write('None')

                elif file == 'shoe':
                    with st.expander('Top 5 recommndations for Shoes'):
                        if len(index_pos_shoes) != 0:

                            columns = st.columns(10)
                            for i in range(len(columns)):
                                with columns[i]:
                                    temp = ' '.join(filenames[index_pos_shoes[i]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    path='/'.join(filenames[similarity[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])

                                    st.image(image)

                        else:
                            st.write('None')
                elif file == 'jacket':
                    with st.expander('Top 5 recommndations for Jacket'):
                        if len(index_pos_jacket) != 0:

                            columns = st.columns(10)
                            for i in range(len(columns)):
                                with columns[i]:
                                    temp = ' '.join(filenames[index_pos_jacket[i]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    path='/'.join(filenames[similarity[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])

                                    st.image(image)

                        else:
                            st.write('None')

    
    
if st.checkbox('Upload a multi-person photo: '):
    uploaded_image = st.file_uploader('Upload an image')
    if uploaded_image is not None:
        
        if save_uploaded_image(uploaded_image):
            if os.path.exists('results'):
                shutil.rmtree('results')
            else:
                os.mkdir('results')

            similarity = []
            for file in os.listdir('uploads/'):

                features = extract('uploads/' + file)

    #         for i in range(len(features_images)):

    #             similarity.append((cosine_similarity(features_images[i].reshape(1, -1), features.reshape(1,-1))))

    #         similarity = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i]
    #         print(similarity)

            # for i in range(5):
            #     st.image('myntradataset/images/' + similarity)
            model.predict(os.path.join('uploads/', uploaded_image.name), save=True,  save_txt=True, save_crop=True, project='results')


    # #         for file in os.listdir('results/predict/crops/'):

    # #             if file in ['shirt', 'jacket', 'dress']:

    # #                 file2 = 'shirt'

    # #                 os.rename('results/predict/crops/{}/'.format(file), 'results/predict/crops/{}/'.format(file2))

            for file in os.listdir('results/predict/crops/'):
                # indices=[]
                # ls=[]
                # print(file)
                similarity = []
                for img in os.listdir('results/predict/crops/{}/'.format(file)):
                    # print(img)

                    # image = Image.open('results/predict/crops/{}/'.format(file)+ img)
                    # image = preprocess('results/predict/crops/{}/'.format(file)+ img)
                    # print(img)

                    # features = resnet50.predict(image).flatten()
                    features=preprocess('results/predict/crops/{}/'.format(file)+ img)
                    # print(features)
                    # distances, indices = nn.kneighbors([features, n_neighbors=5])
                    # for i in range(len(features_images)):

                    # image = Image.open('results_hed/predict/crops/{}/'.format(file)+ img)
                    # print(img)

                    # features = vgg16.predict(image).flatten()
                    # print(features)
                    for i in range(len(features_images)):

                        similarity.append((cosine_similarity(features_images[i].reshape(1, -1), features.reshape(1,-1))))
                        # print(features)
                    # print(similarity)

                    # index_pos = 
    #             # index_pos_'{}'.format(file) = []

                    # indices.append()
                        # print(features)
                    # print(similarity)
                    # for i in range(len(features_images)):

                        # similarity.append((cosine_similarity(features_images[i].reshape(1, -1), features.reshape(1,-1))))
                    # print(indices)
                    # print(distances)
                # print(indices[0][0])
                    # print(index_pos)
                # for i in range(5):
                    # index_pos_'{}'.format(file) = []
                    # print(file)
                    if file == 'shirt':
                        for i in range(5):
                            index_pos_shirts.append(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i][0])
                        # print(distances)

                    elif file == 'shorts':
                        for i in range(5):
                            index_pos_shorts.append(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i][0])
                        # print(distances)

                    elif file == 'pants':
                        for i in range(5):
                            index_pos_pants.append(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i][0])
                        # print(distances)


                    elif file == 'shoe':
                        for i in range(5):
                            index_pos_shoes.append(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i][0])
                        # print(distances)
                    elif file == 'jacket':
                        for i in range(5):
                            index_pos_jacket.append(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[i][0])

            print(index_pos_pants)
            print(index_pos_shirts)  
            print(index_pos_shoes)  
            print(index_pos_shorts)  
            # if st.button('Show'):
            for file in os.listdir('results/predict/crops/'):
                if file == 'short':
                    with st.expander('Top 5 recommndations for short'):
                        if len(index_pos_shorts) != 0:

                            columns = st.columns(5)
                            for i in range(len(columns)):
                                with columns[i]:
                                    # homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                    # print(recommendations[i])
                                    image = cv2.imread('images/{}'.format(filenames[index_pos_shorts[i]]))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_short[i],4])

                                    st.image(image)
                                    # url = "https://www.streamlit.io"
                                    # st.write("[Explore](%s)" % homepage_url)

                        else:
                            st.write('None')

                elif file == 'pants':
                    with st.expander('Top 5 recommndations for Pants'):
                        if len(index_pos_pants) != 0:

                            columns = st.columns(5)
                            for i in range(len(columns)):
                                with columns[i]:
                                    # homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                    # print(recommendations[i])
                                    image = cv2.imread('images/{}'.format(filenames[index_pos_pants[i]]))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_pants[i],4])

                                    st.image(image)
                                    # url = "https://www.streamlit.io"
                                    # st.write("[Explore](%s)" % homepage_url)

                        else:
                            st.write('None')

                elif file == 'shirt':
                    with st.expander('Top 5 recommndations for Shirts'):
                        if len(index_pos_shirts) != 0:


                            columns = st.columns(5)
                            for i in range(len(columns)):
                                with columns[i]:
                                    # homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                    # print(recommendations[i])
                                    image = cv2.imread('images/{}'.format(filenames[index_pos_shirts[i]]))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])

                                    st.image(image)
                                    # url = "https://www.streamlit.io"
                                    # st.write("[Explore](%s)" % homepage_url)


                        else:
                            st.write('None')

                elif file == 'shoe':
                    with st.expander('Top 5 recommndations for Shoes'):
                        if len(index_pos_shoes) != 0:

                            columns = st.columns(5)
                            for i in range(len(columns)):
                                with columns[i]:
                                    # homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                    # print(recommendations[i])
                                    image = cv2.imread('images/{}'.format(filenames[index_pos_shoes[i]]))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shoes[i],4])

                                    st.image(image)
                                    # url = "https://www.streamlit.io"
                                    # st.write("[Explore](%s)" % homepage_url)

                        else:
                            st.write('None')
                elif file == 'jacket':
                    with st.expander('Top 5 recommndations for Jacket'):
                        if len(index_pos_jacket) != 0:

                            columns = st.columns(5)
                            for i in range(len(columns)):
                                with columns[i]:
                                    # homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                    # print(recommendations[i])
                                    image = cv2.imread('images/{}'.format(filenames[index_pos_jacket[i]]))
                                    # image = cv2.resize(image, (224, 224))
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shoes[i],4])

                                    st.image(image)
                                    # url = "https://www.streamlit.io"
                                    # st.write("[Explore](%s)" % homepage_url)

                        else:
                            st.write('None')

