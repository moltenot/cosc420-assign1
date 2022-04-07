import os
import numpy as np
import tensorflow as tf
from make_numpy_dataset import make_dataset
from utils import shuffle_and_split, make_callbacks
from models import make_alexnet_race_model


DATA_DIR = './train'
EPOCHS = 400
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.8
PATIENCE = 50
CHECKPOINT_PATH = 'race-ckpt/alexnetlike-1/cp-{epoch:04d}.ckpt'
CHECKPOINT_DIR = os.dirname(CHECKPOINT_PATH)
TFBOARD_DIR = 'race-logs'


if not (os.path.exists('images.npy') and os.path.exists('races.npy')):
    make_dataset(DATA_DIR)

images = np.load('images.npy')
races = np.load('races.npy')

# turn the numpy dataset into a tensorflow one
dataset = tf.data.Dataset.from_tensor_slices((images, races))
train_dataset, test_dataset = shuffle_and_split(
    dataset, BATCH_SIZE, TRAIN_TEST_SPLIT)

# make the model
model = make_alexnet_race_model()

# callbacks
checkpoint_callback, early_stopping_callback, tensorboard_callback = make_callbacks(
    PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

train_history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=[checkpoint_callback,
               tensorboard_callback, early_stopping_callback]
)
model.save_weights(os.path.join(CHECKPOINT_DIR, 'model_at_stop.ckpt'))
