from keras.applications import inception_v3
from keras.preprocessing import image
import numpy as np

model = inception_v3.InceptionV3()

img = image.load_img("img/sigiriya/300px-Sigiriya.jpg", target_size=(299,299))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = inception_v3.preprocess_input(x)

pred = model.predict(x)

print(model.summary())

classes = inception_v3.decode_predictions(pred, top=5)

print(classes)

