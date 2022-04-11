import os
import tensorflow as tf
from make_numpy_dataset import make_dataset
from utils import get_dataset, make_callbacks, get_settings
from models import make_alexnet_gender_model, make_vgg_gender_model

# change this with each iteration
# the weights will be stored here under subdir 'ckpt' and tensorboard logs under 'logs'
MODEL_PATH = 'gender/vgg-3'  

# get these settings, which are share among the age, race and gender models
DATA_DIR, EPOCHS, BATCH_SIZE, TRAIN_TEST_SPLIT, PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR = get_settings(
    MODEL_PATH)


# check if the dataset has been created yet, if not, create it
if not (os.path.exists('images.npy') and os.path.exists('genders.npy')):
    make_dataset(DATA_DIR)

# get the training and validation datasets. These will be augmented with horizontal
# flipping and some rotation
train_dataset, test_dataset = get_dataset(
    'gender', BATCH_SIZE, TRAIN_TEST_SPLIT, vgg=True)

model = make_vgg_gender_model()

# callbacks
checkpoint_callback, early_stopping_callback, tensorboard_callback = make_callbacks(
    PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

train_history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=[checkpoint_callback,
               tensorboard_callback,
               early_stopping_callback]
)
