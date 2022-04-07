import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from make_numpy_dataset import make_dataset
from utils import IMAGE_SIZE
from models import make_alexnet_gender_model

DATA_DIR = './train'
EPOCHS=400
BATCH_SIZE = 32
TRAIN_TEST_SPLIT=0.8
PATIENCE=50
CHECKPOINT_PATH='gender-ckpt/alexnetlike-1/cp-{epoch:04d}.ckpt'
CHECKPOINT_DIR=os.dirname(CHECKPOINT_PATH)
TFBOARD_DIR='gender-logs'

# check if the dataset has been created yet
if not (os.path.exists('images.npy') and os.path.exists('gender.npy')):
    make_dataset(DATA_DIR)

images =np.load('images.npy')
genders=np.load('genders.npy')

# turn the numpy dataset into a tensorflow one
dataset = tf.data.Dataset.from_tensor_slices((images, genders))
dataset = dataset.shuffle(buffer_size=100).batch(BATCH_SIZE)

# split into training and testing
number_for_training = int(dataset.cardinality().numpy() * TRAIN_TEST_SPLIT)
train_dataset=dataset.take(number_for_training)
test_dataset=dataset.skip(number_for_training)

print(f'train_dataset={train_dataset}')
print(f'test_dataset={test_dataset}')


model=make_alexnet_gender_model()
print(model.summary())

## callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True)
early_stopping_callback =  tf.keras.callbacks.EarlyStopping( monitor='val_accuracy' , patience=PATIENCE )
tensorboard_callback = tf.keras.callbacks.TensorBoard( log_dir=TFBOARD_DIR)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

train_history=model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=[checkpoint_callback,tensorboard_callback,early_stopping_callback]
)
model.save_weights(os.path.join(CHECKPOINT_DIR, 'model_at_stop.ckpt'))