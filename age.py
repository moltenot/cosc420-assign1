import os
import numpy as np
import tensorflow as tf
from make_numpy_dataset import make_dataset
from utils import shuffle_and_split, make_callbacks
from models import make_alexnet_age_model

DATA_DIR = './train'
EPOCHS = 400
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.8
PATIENCE = 30
CHECKPOINT_PATH = 'age-ckpt/alexnetlike-2/cp-{epoch:04d}.ckpt'
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
TFBOARD_DIR = 'age-logs'



def main():
    # check if the dataset has been created yet
    if not (os.path.exists('images.npy') and os.path.exists('ages.npy')):
        make_dataset(DATA_DIR)

    images = np.load('images.npy')
    ages = np.load('ages.npy')

    # turn the numpy dataset into a tensorflow one
    dataset = tf.data.Dataset.from_tensor_slices((images, ages))
    train_dataset, test_dataset = shuffle_and_split(
        dataset, BATCH_SIZE, TRAIN_TEST_SPLIT, augment=True)

    model = make_alexnet_age_model()
    print(model.summary())

    # callbacks
    checkpoint_callback, early_stopping_callback, tensorboard_callback = make_callbacks(
        PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR)

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )

    train_history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[checkpoint_callback,
                   tensorboard_callback, early_stopping_callback]
    )


if __name__ == '__main__':
    main()
