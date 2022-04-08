import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models import make_alexnet_gender_model, make_alexnet_age_model, make_alexnet_race_model
from utils import get_image_data_from_file

#################### constants ####################
GENDER_DIR = 'gender-ckpt/alexnetlike-2'
# GENDER_WEIGHTS=tf.train.latest_checkpoint(GENDER_DIR)
GENDER_WEIGHTS = 'gender-ckpt/alexnetlike-2/cp-0135.ckpt'

AGE_DIR = 'age-ckpt/alexnetlike-2'
AGE_WEIGHTS = tf.train.latest_checkpoint(AGE_DIR)

RACE_DIR = 'race-ckpt/alexnetlike-2'
RACE_WEIGHTS = tf.train.latest_checkpoint(RACE_DIR)

DIR = './test'
files = [os.path.join(DIR, f) for f in os.listdir(DIR)]

print(f'GENDER_WEIGHTS={GENDER_WEIGHTS}')
print(f'AGE_WEIGHTS={AGE_WEIGHTS}')
print(f'RACE_WEIGHTS={RACE_WEIGHTS}')


#################### load and predict ####################
images = [get_image_data_from_file(f) for f in files]
x = tf.convert_to_tensor(images)
print(f'x.shape={x.shape}')

print('loading the gender model')
gender_model = make_alexnet_gender_model()
gender_model.load_weights(GENDER_WEIGHTS)
gender_pred = [['male', 'female']
               [np.argmax(p)] for p in gender_model.predict(x)]
print(f'gender_pred={gender_pred}')

print('loading age model')
age_model = make_alexnet_age_model()
age_model.load_weights(AGE_WEIGHTS)
age_pred = [round(i[0]) for i in age_model.predict(x)]
print(f'age_pred={age_pred}')

print('loading the race model')
race_model = make_alexnet_race_model()
race_model.load_weights(RACE_WEIGHTS)
race_pred = [['white', 'black', 'asian', 'indian', 'others']
             [np.argmax(p)] for p in race_model.predict(x)]
print(f'race_pred={race_pred}')


#################### display ####################
for image, gender, age, race in zip(images, gender_pred, age_pred, race_pred):
    plt.imshow(image)
    plt.title(
        f"predicted gender: {gender}\npredicted age: {age}\npredicted race: {race}")
    plt.show()
