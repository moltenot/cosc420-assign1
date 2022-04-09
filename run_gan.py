from matplotlib import pyplot as plt
import tensorflow as tf
from models import make_basic_generator_model


generator = make_basic_generator_model()

checkpoint_dir = './gan-checkpoints/run-2'
checkpoint = tf.train.Checkpoint(generator=generator)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

num_examples = 3
seed = tf.random.normal([num_examples, 100])# 100 is the input dim of the generator

images = generator(seed)

for image in images:
    plt.imshow(image)
    plt.show()
