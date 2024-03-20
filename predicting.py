import joblib
import streamlit as st
import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
#from skimage.filters import threshold_otsu


def preprocess_image(image):
    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # Resize to 28x28 pixels and normalize pixel values [0,1]
    resized_img = cv2.resize(gray_img, (28, 28))   
    normalized_img = resized_img / 255.0

    # Invert pixel vaues due to MNIST data being inverted   
    inverted_img = (1 - normalized_img) * 255.0   

    # Apply thresholding, setting pixels below mean to black
    mean_pixel_value = np.mean(inverted_img)   
    inverted_img[inverted_img < mean_pixel_value] = 0 
   
    # Reshape, 1d-array and then 2d-array (1, 784)
    flattened_img = inverted_img.flatten()   
    reshaped_img_2d = flattened_img.reshape(1, -1)

    # Load and apply scaler
    scaler = joblib.load("scaler.pkl")
    img_ready = scaler.transform(reshaped_img_2d)

    return img_ready.flatten()

def make_prediction(image):
    model = joblib.load("C:/Users/tovat/OneDrive/Dokument/EC_utbildning/MachineLearning/final_model.pkl")
    predicted_number = model.predict([image])

    return predicted_number[0]