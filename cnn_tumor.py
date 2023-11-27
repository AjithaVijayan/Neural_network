import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

image_dir = r"DL-ALGORITHMS\CNN\tumor_detection\tumordataset\tumordata"
no_tumor_images = os.listdir(image_dir + '/no')
yes_tumor_images = os.listdir(image_dir + '/yes')

dataset = []
label = []
img_siz = (128, 128)

for i, image_name in tqdm(enumerate(no_tumor_images), desc="No Tumor"):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_dir + '/no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in tqdm(enumerate(yes_tumor_images), desc="Tumor"):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_dir + '/yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

model.save("tumor_detection_model.h5")

with open("tumor_detection_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)


