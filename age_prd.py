import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

def min_max_scaling(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    zero_range_indices = np.where(X_max - X_min == 0)
    X_max[zero_range_indices] = 1
    X_scaled = (X - X_min) / (X_max - X_min)
    return X_scaled

def predict_age_gender(image, model):
    resized_image = cv2.resize(image, (100, 100))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.reshape(100, 100, 1)
    input_image = np.expand_dims(gray_image, axis=-1)
    input_image = min_max_scaling(input_image)
    age = model.predict(np.array([input_image]))
    return age[0][0]

model = load_model('age_model_finalrss.h5')


st.header("Age Detection")
img_file_buffer = st.file_uploader('Upload a PNG image',  type=['png', 'jpg'])

if img_file_buffer is not None:

    image = cv2.imdecode(np.frombuffer(img_file_buffer.read(), np.uint8), 1)
    resized_image = cv2.resize(image, (200, 200))

    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    st.image(rgb_image, caption='Image Uploaded.', use_column_width=True)

    if st.button("Predict"):
        age = predict_age_gender(image, model)
        st.write("Predicted age:", round(age,0))