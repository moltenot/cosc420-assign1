from tensorflow import keras
import os
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import parse_image_for_gender,imagesize

datadir = 'UTKFace'

dataset = tf.data.Dataset.list_files('data/UTKFace/*', shuffle=True)
dataset = dataset.map(parse_image_for_gender)

"""- age is just encoded as a number
- gender is a binary 0 or 1
- race is a one-hot encoded vector, with dimension 5
- image is a 3 channel image of size `imagesize`
"""

number_for_training = int(dataset.cardinality().numpy() * 0.8)
train_dataset = dataset.take(number_for_training)
test_dataset = dataset.skip(number_for_training)

"""## Defining the Model
removed softmax, that might have been messing stuff up
"""

# copied and adjusted from https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
# changed model input to fit my (smaller) images
# changed output for gender
# kept all the layers


model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(
        4, 4), activation='relu', input_shape=imagesize + [3]),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=256, kernel_size=(
        5, 5), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=384, kernel_size=(
        3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(
        3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])
print(model.summary())

"""## Compiling the Model

mostly following [this example](https://colab.research.google.com/github/shubham0204/Google_Colab_Notebooks/blob/main/Gender_Estimation_(W2).ipynb#scrollTo=cow71DK6kjUQ )

### Callbacks
- modelCheckpoint
- TensorBoard
- EarlyStopping
"""

batch_size = 32
num_epochs = 50
save_dir = 'train-gender-3/cp.ckpt'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(save_dir)

logdir = os.path.join(
    "tb_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=3)


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

"""## Fitting the Model

"""

train_history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=test_dataset,
    callbacks=[checkpoint_callback,
               tensorboard_callback, early_stopping_callback]
)

"""## Evaluating the Model"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir tb_logs

os.listdir(save_dir)
model.load_weights(save_dir)
loss, acc = model.evaluate(test_dataset, verbose=2)
