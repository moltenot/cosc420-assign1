import numpy as np
import tensorflow as tf

IMAGE_SIZE = [100, 100]


def parse_image(filename):
    im = get_image_data_from_file(filename)

    # parse the filename
    parts = tf.strings.split(tf.strings.split(filename, '/')[2], '_')
    age = tf.strings.to_number(parts[0])
    gender = tf.one_hot(tf.strings.to_number(parts[1],
                                             out_type=tf.dtypes.int32), 2)
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
    return ages, images, races, genders


def shuffle_and_split(dataset, BATCH_SIZE, TRAIN_TEST_SPLIT):
    dataset = dataset.shuffle(buffer_size=100).batch(BATCH_SIZE)

    # split into training and testing
    number_for_training = int(dataset.cardinality().numpy() * TRAIN_TEST_SPLIT)
    train_dataset = dataset.take(number_for_training)
    test_dataset = dataset.skip(number_for_training)
    return train_dataset, test_dataset


def make_callbacks(PATIENCE, CHECKPOINT_PATH, TFBOARD_DIR):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=PATIENCE)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TFBOARD_DIR)
    return checkpoint_callback, early_stopping_callback, tensorboard_callback
