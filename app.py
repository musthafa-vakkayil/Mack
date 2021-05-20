import streamlit as st
import keras
from PIL import Image, ImageOps
import numpy as np


st.header("Mango or JackFruit Classifier")
st.text("Please upload an image of mango or jackfruit")

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability

uploaded_file = st.file_uploader("Choose an image ...", type=["jpg","jpeg"])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'model.h5')
        if label == 0:
            st.markdown('This image is most likely:Mango')
        else:
            st.markdown('This image is most likely:JackFruit')