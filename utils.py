import tensorflow as tf

IMAGE_SIZE=[128,128]

def parse_image(filename):
    image_raw = tf.io.read_file(filename)
    im = tf.image.decode_jpeg(image_raw, channels=3)
    im = tf.image.resize(im, IMAGE_SIZE) / 255

    # parse the filename
    parts = tf.strings.split(tf.strings.split(filename, '/')[2], '_')
    age = tf.strings.to_number(parts[0])
    gender = tf.strings.to_number(parts[1])
    race = tf.one_hot(tf.strings.to_number(
        parts[2], out_type=tf.dtypes.int32), 5)

    return age, gender, race, im