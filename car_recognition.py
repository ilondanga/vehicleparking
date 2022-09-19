import tensorflow as tf
import numpy as np
import keras.preprocessing as kr
import keras
import keras_preprocessing
# from vehicle_parking import path

height=180
width=180

class_names=['free','full']
model=keras.models.load_model('vehicle_model.h5')

path="C:/Users/patricia/Desktop/view7.png"
img=kr.image.load_img(path,target_size=(height,width))
img=kr.image.img_to_array(img)
img=tf.expand_dims(img,0)
print(img.shape)

predictions=model.predict(img)

score=tf.nn.softmax(predictions[0])
print("This parking space is {} with a{:.2f} percent confidence."
      .format(class_names[np.argmax(score)],100*np.max(score)))
