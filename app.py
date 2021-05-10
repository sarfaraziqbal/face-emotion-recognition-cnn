import av
import cv2
import numpy as np
import streamlit as st        
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer     #streamlit-webrtc helps to deal with real-time video streams.

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers


my_model = tf.keras.models.load_model('../input/model5/model.h5')   
class VideoTransformer(VideoTransformerBase): 
    def transform(self, frame):                                                            
        img = frame.to_ndarray(format="bgr24")                                              
        face_detect = cv2.CascadeClassifier('../input/model56/haarcascade_frontalface_default.xml')          
        #eye_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
        
        class_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']        



        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                 
        face_roi = face_detect.detectMultiScale(img_gray, 1.3,1)          
        if face_roi is ():                                                
            return img

        for(x,y,w,h) in face_roi:                                          
            x = x - 5
            w = w + 10
            y = y + 7
            h = h + 2
            cv2.rectangle(img, (x,y),(x+w,y+h),(125,125,10), 2)             
            img_color_crop = img[y:y+h,x:x+w]                             
            final_image = cv2.resize(img_color_crop, (48,48))          
            final_image = np.expand_dims(final_image, axis = 0)           
            final_image = final_image/255.0                               
            prediction = my_model.predict(final_image)                    
            label=class_labels[prediction.argmax()]                        
            cv2.putText(img,label, (50,60), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2, (120,10,200),3)    
                                                     
                                                      
                                                      
       
        return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)      