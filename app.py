import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

from predicting import make_prediction, preprocess_image


# Change layout and interface
nav = st.sidebar.radio("Navigation Menu",["Main", "Upload photo", "Capture photo"])

if nav == "Main":
    st.title('Predict Handwritten Digits')
    st.markdown('Welcome to my Handwritten Digit Prediction Model!')
    st.markdown('Upload a photo of a handwritten digit or capture one directly using your webcam.')
                

if nav == "Upload photo":
    st.title('Upload photo')
    st.markdown('Upload a photo of a handwritten digit and let the model predict it for you')

    uploaded_image = st.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

    # Prediction
    if st.button('Predict'):
        if uploaded_image is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
            processed_image = preprocess_image(image)
            predicted_digit = make_prediction(processed_image) 
            processed_image = processed_image.astype(np.float32)
            processed_image = np.clip(processed_image, 0.0, 1.0)
            processed_image_resized = cv2.resize(processed_image.reshape(28, 28), (image.shape[1], image.shape[0])) 
            
            # Display the original and processed images
            st.image([image, processed_image_resized], caption=['Original photo', 'Processed photo'], width=200)
            # Display prediction
            st.write(f'### Predicted digit: {predicted_digit}')

        else:
            st.warning('Please upload photo to make a prediction')

if nav == "Capture photo":
    st.title('Capture photo')
    st.markdown('Take a photo of a handwritten digit with your webcam and let the model predict it for you')
    st.markdown('<p style="color:red;">Please ensure good lighting and try to keep the digit centered in the frame.</p>', unsafe_allow_html=True)
    captured_image = st.camera_input('', disabled=False)

    # Prediction
    if st.button('Predict'):
        if captured_image is not None:
            image = cv2.imdecode(np.frombuffer(captured_image.read(), np.uint8), 1)
            processed_image = preprocess_image(image)
            predicted_digit = make_prediction(processed_image) 
            processed_image = processed_image.astype(np.float32)
            processed_image = np.clip(processed_image, 0.0, 1.0)
            processed_image_resized = cv2.resize(processed_image.reshape(28, 28), (image.shape[1], image.shape[0])) 

            # Display the original and processed images
            st.image([image, processed_image_resized], caption=['Original Image', 'Processed Image'], width=200)
            # Display prediction
            st.write(f'### Predicted digit: {predicted_digit}')

        else:
            st.warning('Please capture photo to make a prediction')