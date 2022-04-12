import os
from make_numpy_dataset import make_dataset
from utils import get_dataset, make_callbacks, get_settings, get_optimizer
from models import make_alexnet_age_model, make_vgg_age_model

# change this with each iteration
# the weights will be stored here under subdir 'ckpt' and tensorboard logs under 'logs'
MODEL_PATH = 'age/alexlike-1-1'

# get these settings, which are share among the age, race and gender models
DATA_DIR, EPOCHS, BATCH_SIZE, TRAIN_TEST_SPLIT, PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR = get_settings(
    MODEL_PATH)


# check if the dataset has been created yet, if not, create it
if not (os.path.exists('images.npy') and os.path.exists('ages.npy')):
    make_dataset(DATA_DIR)

# get the training and validation datasets. These will be augmented with horizontal
# flipping and some rotation
train_dataset, test_dataset = get_dataset(
    'age', BATCH_SIZE, TRAIN_TEST_SPLIT)

model = make_alexnet_age_model()

# callbacks
checkpoint_callback, early_stopping_callback, tensorboard_callback = make_callbacks(
    PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR, metric='val_loss')

model.compile(
    optimizer=get_optimizer(),
    loss='mse',
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
