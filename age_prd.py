import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from PIL import Image


custom_objects = {
    'mse': MeanSquaredError(),
}

model_path = 'D:/DeepLearning/age_gen_pred/age_gen_fin.h5'
model = load_model(model_path, custom_objects=custom_objects)

def preprocess_image(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((128, 128))
    img_array = np.array(image)
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img = gray_img.reshape(1, 128, 128, 1)
    img = img.astype('float32') / 255.0
    return img, img_array

def predict_age_gender(model, image):
    img, plt_img = preprocess_image(image)
    pred = model.predict(img)
    pred_gender = "Male" if round(pred[0][0][0]) == 0 else "Female"
    pred_age = round(pred[1][0][0])
    return pred_gender, pred_age, plt_img

st.title("Age and Gender Predictor")
st.write("Upload an image to predict the age and gender.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    thumbnail = image.resize((300, 300))
    st.image(thumbnail, caption='Uploaded Image', use_column_width=False)

    if st.button('Predict'):
        gender, age, plt_img = predict_age_gender(model, uploaded_file)
        st.write(f"Predicted Gender: {gender}")
        st.write(f"Predicted Age: {age}")
