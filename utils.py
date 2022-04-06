import numpy as np
import tensorflow as tf

IMAGE_SIZE=[100, 100]

def parse_image(filename):
    im = get_image_data_from_file(filename)

    # parse the filename
    parts = tf.strings.split(tf.strings.split(filename, '/')[2], '_')
    age = tf.strings.to_number(parts[0])
    gender = tf.strings.to_number(parts[1])
    race = tf.one_hot(tf.strings.to_number(
        parts[2], out_type=tf.dtypes.int32), 5)

    return age, gender, race, im

def get_image_data_from_file(filename):
    image_raw = tf.io.read_file(filename)
    im = tf.image.decode_jpeg(image_raw, channels=3)
    im = tf.image.resize(im, IMAGE_SIZE) / 255
    return im

def get_data_from_imagepaths(image_paths):
    ages = []
    images = []
    races = []
    genders = []
    count = 0
    for im in image_paths:
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
    return ages,images,races,genders