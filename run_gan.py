import os
from matplotlib import pyplot as plt
import tensorflow as tf
from models import make_basic_generator_model


#################### get the model ####################
generator = make_basic_generator_model()

checkpoint_dir = './gan-checkpoints/run-3'
checkpoint = tf.train.Checkpoint(generator=generator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#################### generate images ####################
num_examples = 10
seed = tf.random.normal([num_examples, 100]) # 100 is the input dim of the generator
images = generator(seed)

#################### save the images ####################

# create the fake subdir if it doesn't exist
if not os.path.exists('fake'):
    os.makedirs('fake')

for i, image in enumerate(images):
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('fake/image_%d.png' % (i+1))
    # plt.show()
