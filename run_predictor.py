import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from age import make_alexnet_age_model
from utils import get_image_data_from_file


DIR = './test'
files = [os.path.join(DIR, f) for f in os.listdir(DIR)]
print(f'files={files}')

images = [get_image_data_from_file(f) for f in files]

print('loading the age model')
age_model=make_alexnet_age_model()

for image in images:
    plt.imshow(image)
    plt.show()
