from genericpath import exists
import os
import numpy as np
import tensorflow as tf
from make_numpy_dataset import make_dataset
from utils import IMAGE_SIZE

DATA_DIR = './train'
EPOCHS=5
BATCH_SIZE = 10
TRAIN_TEST_SPLIT=0.8

# check if the dataset has been created yet
if not (os.path.exists('images.npy') and os.path.exists('ages.npy')):
    make_dataset(DATA_DIR)

images =np.load('images.npy')
ages=np.load('ages.npy')

# turn the numpy dataset into a tensorflow one
dataset = tf.data.Dataset.from_tensor_slices((images, ages))
dataset = dataset.shuffle(buffer_size=100).batch(BATCH_SIZE)

# split into training and testing
number_for_training = int(dataset.cardinality().numpy() * TRAIN_TEST_SPLIT)
train_dataset=dataset.take(number_for_training)
test_dataset=dataset.skip(number_for_training)

print(f'train_dataset={train_dataset}')
print(f'test_dataset={test_dataset}')


from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=IMAGE_SIZE + [3]),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])
print(model.summary())


model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

train_history=model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
)