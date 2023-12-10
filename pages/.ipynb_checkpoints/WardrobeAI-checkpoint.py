import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras import Sequential

from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, Flatten
import cv2
import os
import random
from tqdm import tqdm
import numpy as np
from keras.preprocessing import image
import os
import random
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st
import shutil

st.markdown("<h1 style='text-align: center; color: white;'>Wardrobe AI</h1>", unsafe_allow_html=True)
st.divider()

model=tf.keras.saving.load_model('clothes (1).keras')



def get_data_single(folder_path):
#     temp = []
    temp = []
    
    features = []
    X = []
   
#         for img in tqdm(os.listdir(folder_path)):
    img = image.load_img(folder_path, target_size=(128, 128, 3))
    img_array = image.img_to_array(img)
    img_array /= 255.
#             resize_array = cv2.resize(img_array, (128,128))
    temp.append(img_array)

#     for i in temp:
#         features.append(i)
#     X = np.array(temp).reshape(-1, 128, 128, 3)
#     X = X.astype('float32')
#     X /= 255
    return np.array(temp).reshape(-1, 128, 128, 3)


if os.path.exists('uploads_shoes/'):
    shutil.rmtree('uploads_shoes/')
    os.mkdir('uploads_shoes/')

else:
    os.mkdir('uploads_shoes/')

    
if os.path.exists('uploads_pants/'):
    shutil.rmtree('uploads_pants/')
    os.mkdir('uploads_pants/')

else:
    os.mkdir('uploads_pants/')
    
    
    
if os.path.exists('uploads_outwear/'):
    shutil.rmtree('uploads_outwear/')
    os.mkdir('uploads_outwear/')

else:
    os.mkdir('uploads_outwear/')


def save_uploaded_image_outwear(uploaded_image):
    for i in uploaded_image:
        # try:
        with open(os.path.join('uploads_outwear',i.name),'wb') as f:
            f.write(i.getbuffer())
            # return True
        # except:
            # return False
    
    
def save_uploaded_image_pants(uploaded_image):
    
    for i in uploaded_image:
        # try:
        with open(os.path.join('uploads_pants',i.name),'wb') as f:
            f.write(i.getbuffer())
            # return True
        # except:
            # return False
    
    
def save_uploaded_image_shoes(uploaded_image):
    
    for i in uploaded_image:
        # try:
        with open(os.path.join('uploads_shoes',i.name),'wb') as f:
            f.write(i.getbuffer())
            # # return True
        # except:
            # return False

outwear = []
pants = []
shoes = []


uploaded_image = st.file_uploader('Upload outwear images',accept_multiple_files=True, key=1)

save_uploaded_image_outwear(uploaded_image)
        
 
        
        
uploaded_image = st.file_uploader('Upload pants images',accept_multiple_files=True, key=2)

save_uploaded_image_pants(uploaded_image)
        


uploaded_image = st.file_uploader('Upload shoes images',accept_multiple_files=True, key=3)

save_uploaded_image_shoes(uploaded_image)
        
        


for file in os.listdir('uploads_outwear/'):
    
     outwear.append(os.path.join('uploads_outwear', file))
        
        
        
for file in os.listdir('uploads_pants/'):
    
     pants.append(os.path.join('uploads_pants', file))
        
        
        
for file in os.listdir('uploads_shoes/'):
    
     shoes.append(os.path.join('uploads_shoes', file))

        
        
print(outwear)
print(pants)
print(shoes)

values = []
clothes = []
outwear_input = get_data_single(outwear[0])
pants_input=get_data_single(pants[0])
shoes_input = get_data_single(shoes[0])

# outwear_input=get_data_single(outwear[0])
# print(outwear_input)



for i in range(3):
    for j in range(3):
        for k in range(3):
            if k==1:
                outwear_input=get_data_single(outwear[i])
                print(outwear_input)
            elif k==2 :
                pants_input=get_data_single(pants[j])
            elif k==3 :
                shoes_input=get_data_single(shoes[k])
            y_pred = model.predict([outwear_input,pants_input,shoes_input])
            values.append(y_pred[0][0])
            clothes.append([outwear[i], pants[j], shoes[k]])
            
            
            
for value in range(len(values)):
    values[value] = round(values[value], 2)
    
    
temp_2 = []


for i in range(len(values)):
    temp_2.append((i,values[i]))
    
    
from operator import itemgetter
temp_2 = sorted(temp_2,key=itemgetter(1), reverse=True)


seen = set()
new = []
for i in range(len(temp_2)):
    if temp_2[i][1] not in seen:
        seen.add(temp_2[i][1])
        new.append(temp_2[i])
        
        
col1, col2, col3  = st.columns([2,2,2])
id_ = 0
if st.checkbox('Predict!'):
    for i in range(len(new)):
        index = new[i][0]
        count = 0
        for j in clothes[index]:
            print(j)
        # print(new[i][1])
    #         array = img.imread(j)
    #         plt.imshow(array)
        # print('-------------------------------------------------------------------------------------------------')
            with st.expander('Show'):
                if count == 0:
                    with col1:
                        array = cv2.imread(j)
                        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                        st.image(rgb)

                if count == 1:
                     with col2:
                        array = cv2.imread(j)
                        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                        st.image(rgb)


                if count == 2:
                     with col3:
                        array = cv2.imread(j)
                        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                        st.image(rgb)
                count+=1
                id_+=1