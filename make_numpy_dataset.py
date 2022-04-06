import os
from utils import get_data_from_imagepaths
import numpy as np


DATA_DIR = './train'

def make_dataset(DATA_DIR):
    image_paths = [os.path.join(DATA_DIR, file)
                   for file in os.listdir(DATA_DIR)]
    ages, images, races, genders = get_data_from_imagepaths(image_paths)

    np.save('images.npy', images)
    np.save('ages.npy', ages)
    np.save('genders.npy', genders)
    np.save('races.npy', races)


if __name__ == '__main__':
    make_dataset(DATA_DIR)
