"""
In this example code,
1. We will get a pre-trained model from Keras
2. Remove the final dense layer that does the prediction
3. Feed our training data into the conv layers and get the output(features for our dense layer)
4. Save our features
"""
from keras.preprocessing import image
from keras.applications import resnet50
import numpy as np
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# include_top_false -> remove the final dense layer when loading the model
# when the top is false we can also change the image size, thus changing the input node size
feature_model = resnet50.ResNet50(weights="imagenet",include_top=False, input_shape=(200, 200, 3))

positive_path = Path('sigiriya')
negative_path = Path('others')

images = []
labels = []

for img in positive_path.glob("*.jpg"):
    img = image.load_img(img)
    image_array = image.img_to_array(img)
    images.append(image_array)
    labels.append(1)

for img in negative_path.glob("*.png"):
    img = image.load_img(img, target_size=(200,200))
    image_array = image.img_to_array(img)
    images.append(image_array)
    labels.append(0)

X = np.array(images)
Y = np.array(labels)

X = resnet50.preprocess_input(X)

X_input_for_dense_model = feature_model.predict(X)

# Create a new model
model = Sequential()
model.add(Flatten(input_shape=X_input_for_dense_model.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam"
)

# Train the model
model.fit(
    X_input_for_dense_model,
    Y,
    epochs=10,
    shuffle=True
)

img = image.load_img("beach.jpg", target_size=(200, 200))

image_array = image.img_to_array(img)
images = np.expand_dims(image_array, axis=0)
images = resnet50.preprocess_input(images)

features = feature_model.predict(images)
results = model.predict(features)

print(results[0][0])