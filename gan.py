# adapted from
# https://www.tensorflow.org/tutorials/generative/dcgan

import tensorflow as tf
import numpy as np
import os
from make_numpy_dataset import make_dataset, make_gan_images
import matplotlib.pyplot as plt
import time
from models import make_basic_discriminator_model,make_basic_generator_model, make_generator_model, make_alexnet_age_model

BATCH_SIZE = 32

big_images=True # set to false if you want 28*28 small images

if big_images:
    discriminator=make_alexnet_age_model()
    generator=make_generator_model()
else:
    discriminator = make_basic_discriminator_model()
    generator = make_basic_generator_model()


if big_images:
    if not os.path.exists('images.npy'):
        make_dataset()

    images = np.load('images.npy')

else:
    # create gan-images if it doesn't exist
    if not os.path.exists('gan-images.npy'):
        make_gan_images()

    # this is a long numpy array of images of shape (28,28,3) with float values
    # in the range [0,1]
    images = np.load('gan-images.npy')


# batch and shuffle images
train_dataset = tf.data.Dataset.from_tensor_slices(
    images).shuffle(20000).batch(BATCH_SIZE)


# make loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# the discrimiator want to predict real images as 1 and fake images as 0
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# generator loss is low when fake_output is close to 1
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# make checkpointing system
checkpoint_dir = './big-gan-2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# training loop
EPOCHS = 400
noise_dim = 1300
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images to see the progression of the model
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    """saves a plot from each epoch. Each image is a 4*4 grid of images
    @param model: the model for that epoch
    @param epoch: the epoch number
    @param test_input: the input to the model, should be the same each time so we can
          see the progression of the model using the same input
    """
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show() # uncomment this if you want interactive (blocked) training


if __name__ == '__main__':
    train(train_dataset, EPOCHS)
