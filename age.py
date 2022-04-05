from tensorflow import keras
import os
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import parse_image_for_age, imagesize

datadir = 'UTKFace'

dataset = tf.data.Dataset.list_files(f"{datadir}/*", shuffle=True)
dataset = dataset.map(parse_image_for_age)

number_for_training = int(dataset.cardinality().numpy() * 0.8)
train_dataset = dataset.take(number_for_training)
test_dataset = dataset.skip(number_for_training)

"""## Defining the Model
For the age I have to change to not use a softmax output, and change the loss function to a regression
"""

# mostly copied from example3 of lech code
model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1))



"""## Compiling the Model

mostly following [this example](https://colab.research.google.com/github/shubham0204/Google_Colab_Notebooks/blob/main/Gender_Estimation_(W2).ipynb#scrollTo=cow71DK6kjUQ )

### Callbacks
- modelCheckpoint
- TensorBoard
- EarlyStopping
"""

batch_size = 32
num_epochs = 50

save_dir = 'train-age-1/cp.ckpt'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(save_dir)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=3)


model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy'])

train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

"""## Fitting the Model

"""

train_history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=test_dataset,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

os.listdir(save_dir)
model.load_weights(save_dir)
loss, acc = model.evaluate(test_dataset, verbose=2)
