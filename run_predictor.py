import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models import make_alexnet_gender_model, make_alexnet_age_model
from utils import get_image_data_from_file


DIR = './test'
files = [os.path.join(DIR, f) for f in os.listdir(DIR)]
print(f'files={files}')

images = [get_image_data_from_file(f) for f in files]
x=tf.convert_to_tensor(images)
print(f'x.shape={x.shape}')

print('loading the gender model')
gender_model=tf.saved_model.load('train-gender-1/cp.ckpt')
print(list(gender_model.signatures.keys()))
gender_model=gender_model.signatures['serving_default']

print('predicting genders')
gender_pred=[['male','female'][np.argmax(p)] for p in gender_model(x)['dense_2']]

for image, gender in zip(images, gender_pred):
    plt.imshow(image)
    plt.title(f"predicted gender: {gender}")
    plt.show()
