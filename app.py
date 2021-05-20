import streamlit as st
from keras.models import model_from_json
#import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model():
  #model=tf.keras.models.load_model('C:\\Users\\musth\\Desktop\\Mack\\model.h5')
  #return model
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  model.load_weights("model.h5")
  #print("Loaded model from disk")
  return model
model = load_model()
st.write("""
            Mango or Jackfruit Classifier
         """
        )
file=st.file_uploader("Please upload an image of mango or jackfruit",type=["jpg"])
#import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):

  size=(64,64)
  image= ImageOps.fit(image_data,size,Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=img[np.newaxis,...]
  prediction=model.predict(img_reshape)
  if prediction[0][0]==1:
    preds='Mango'
  else:
    preds='Jackfruit'
  return preds
  #return prediction


if file is None:
  st.text("Please upload an image file")
else:
  image=Image.open(file)
  st.image(image,use_column_width=True)
  predictions = import_and_predict(image,model)
  #class_names=['Mango','Jackfruit']
  string="This image is most likely:"+predictions
  st.success(string)