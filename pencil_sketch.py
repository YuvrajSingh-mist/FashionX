import cv2
import os
from tqdm import tqdm
import gc

i='skirt'

if os.path.exists('pencil_sketches/{}'.format(i)) == False:
    os.mkdir('pencil_sketches/{}'.format(i))

for img in tqdm(os.listdir('myntradataset/images/')):
    image = cv2.imread('myntradataset/images/' + img)
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_not = cv2.bitwise_not(grayimage)
    blur = cv2.GaussianBlur(gray_not, (21,21), 0)
    invert_blur = cv2.bitwise_not(blur)
    
    sketch= cv2.divide(grayimage, invert_blur, scale=255.0) 
    
    cv2.imwrite('pencil_sketches/' + img, sketch)
    gc.collect()