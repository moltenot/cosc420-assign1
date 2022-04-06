import os
from utils import parse_image
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = './train'
EPOCHS = 5
BATCH_SIZE = 10
TRAIN_TEST_SPLIT = 0.8


def make_dataset(DATA_DIR):
    image_paths = [os.path.join(DATA_DIR, file)
                   for file in os.listdir(DATA_DIR)]
    ages = []
    images = []
    races = []
    genders = []
    count = 0
    for im in image_paths[:1000]:
        print(f"\rprocessing image {count}", end="")
        age, gender, race, image = parse_image(im)
        races.append(race)
        genders.append(gender)
        ages.append(age)
        images.append(image)
        count += 1
    print("")

    ages = np.asarray(ages)
    images = np.asarray(images)
    races = np.asarray(races)
    genders = np.asarray(genders)

    print(f'ages.shape={ages.shape}')
    print(f'images.shape={images.shape}')
    print(f'races.shape={races.shape}')
    print(f'genders.shape={genders.shape}')

    np.save('images.npy', images)
    np.save('ages.npy', ages)
    np.save('genders.npy', genders)
    np.save('races.npy', races)


if __name__ == '__main__':
    make_dataset(DATA_DIR)
