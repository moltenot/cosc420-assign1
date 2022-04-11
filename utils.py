import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_SIZE = [100, 100]


def parse_image(filename):
    """returns the image and label data for a specific filepath
    @param filename - String file to parse

    @returns age, gender, race, image - all as tensors
    """
    im = get_image_data_from_file(filename)

    # parse the filename
    parts = tf.strings.split(tf.strings.split(filename, '/')[2], '_')

    age = tf.strings.to_number(parts[0])
    gender = tf.one_hot(tf.strings.to_number(parts[1],
                                             out_type=tf.dtypes.int32), 2)
    race = tf.one_hot(tf.strings.to_number(
        parts[2], out_type=tf.dtypes.int32), 5)

    return age, gender, race, im


def get_image_data_from_file(filename, image_size=IMAGE_SIZE):
    """Parse specifically the image data, resizing to the image_size parameter
    This will rescale the image to have float pixels in the range [0,1]
    @param filename - String - file to parse
    @param image_size - [int, int] - size of the image data to return

    @returns tensor for the image of shape (image_size[0], image_size[1], 3)
    """
    image_raw = tf.io.read_file(filename)
    im = tf.image.decode_jpeg(image_raw, channels=3)
    im = tf.image.resize(im, image_size) / 255
    return im


def get_data_from_imagepaths(image_paths):
    """get label and image data from a list of imagepaths
    @param - image_paths - string[] - list of image paths

    @returns age, images, races, genders
        age - np array of shape (None, )
        image - np array of shape (None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        races - np array of shape (None, 5)
        genders - np array of shape (None, 2)
    """
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
    return ages, images, races, genders


def get_settings(MODEL_PATH):
    """centralized some of the hyperparameters for age, gender and race training
    @param - MODEL_PATH - String-  the path to store checkpoints and tensorboard events in

    @returns - DATA_DIR, EPOCHS, BATCH_SIZE, TRAIN_TEST_SPLIT, PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR
        which are to be used in training each model
    """
    DATA_DIR = './train'
    EPOCHS = 400
    BATCH_SIZE = 80
    TRAIN_TEST_SPLIT = 0.8
    PATIENCE = 20
    CHECKPOINT_DIR = os.path.join(MODEL_PATH, 'ckpt')
    CHECKPOINT_PATH = CHECKPOINT_DIR + "/cp-{epoch:04d}.ckpt"
    TFBOARD_DIR = os.path.join(MODEL_PATH, 'logs')
    return DATA_DIR, EPOCHS, BATCH_SIZE, TRAIN_TEST_SPLIT, PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR


def get_dataset(which_ds, batch_size, percent_train, vgg=False):
    """returns datasets with augmented images object for the training and validation data
    @param which_ds - String - either 'age' or 'gender' or 'race'
    @param batch_size - Int - how big to batch the datasets
    @param percent_train - Float - how much to keep for the training set. 
        i.e. 0.8 would be 20% validation and 80% training
    @param vgg=False - Boolean - set to true to run vgg preprocessing on the images

    @returns train_set, val_set
        train_set - tf.data.Dataset object for training
        val_set - tf.data.Dataset object for validation
    """

    # I need two datagenerators since for the age model, the imageDataGenerator
    # requires that the unique classes in the validation and training label sets
    # be identical. This was not the case since I guess there is not 116 year olds
    # so it would break
    # to deal with this I am using sklearn to split and shufle the data instead
    datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
    )
    
    # don't have augmentation on the validation data https://stackoverflow.com/a/48031128
    datagen_test = tf.keras.preprocessing.image.ImageDataGenerator()

    if vgg:
        images = np.load('vgg-images.npy')
    else:
        images = np.load('images.npy')

    if which_ds == 'gender':
        labels = np.load('genders.npy')
    elif which_ds == 'race':
        labels = np.load('races.npy')
    elif which_ds == 'age':
        labels = np.round(np.load('ages.npy')).astype(int)

    # shuffle and split
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=1-percent_train, random_state=47
    )

    datagen_train.fit(x_train)
    datagen_test.fit(x_test)

    train_set = datagen_train.flow(
        x_train, y_train, batch_size=batch_size)
    val_set = datagen_test.flow(
        x_test, y_test, batch_size=batch_size)

    return train_set, val_set

def get_optimizer():
    """returns the adam optimizer to use for the age, gender and race models
    """
    return tf.keras.optimizers.Adam(learning_rate=1e-4)

def make_callbacks(PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR, metric='val_accuracy'):
    """
    Make the callbacks for to be attached to a model
    @param PATIENCE - the patitence for the early stopping callback
    @param CHECKPOINT_PATH - the path format for checkpoint saving
    @param TFBOARD_DIR - where to save the tensorboard events
    @param metric='val_accuracy' - what to check early stopping for, or saving checkpoints
        since we only save checkpoints if they are better than before based on this metric

    @returns checkpoint_callback, early_stopping_callback, tensorboard_callback 
    """
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=True,
        monitor=metric,
        save_best_only=True)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=metric, patience=PATIENCE)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TFBOARD_DIR)
    return checkpoint_callback, early_stopping_callback, tensorboard_callback
