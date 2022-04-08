import os
from utils import get_data_from_imagepaths, get_image_data_from_file
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

# makes images that are much smaller, just for the GAN
def make_gan_images(image_size=[28, 28]): 
    image_paths=[os.path.join(DATA_DIR, file)
                   for file in os.listdir(DATA_DIR)]
    count=0             
    images=[]
    for im_path in image_paths:
        print(f'\rimage number {count}', end="")
        images.append(get_image_data_from_file(im_path,image_size=image_size))
        count+=1
    np.save('gan-images.npy', np.array(images))

if __name__ == '__main__':
    make_dataset(DATA_DIR)
