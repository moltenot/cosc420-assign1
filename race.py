import os
import numpy as np
import tensorflow as tf
from make_numpy_dataset import make_dataset
from utils import get_dataset, make_callbacks, get_settings
from models import make_alexnet_race_model

MODEL_PATH = 'race/alexnetlike-3'  # change this with each iteration

DATA_DIR, EPOCHS, BATCH_SIZE, TRAIN_TEST_SPLIT, PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR = get_settings(
    MODEL_PATH)


if not (os.path.exists('images.npy') and os.path.exists('races.npy')):
    make_dataset(DATA_DIR)

train_dataset, test_dataset = get_dataset('race', BATCH_SIZE, TRAIN_TEST_SPLIT)

# make the model
model = make_alexnet_race_model()

# callbacks
checkpoint_callback, early_stopping_callback, tensorboard_callback = make_callbacks(
    PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
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
