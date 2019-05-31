from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.callbacks import TensorBoard
import glob
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
import os
import shutil

import sys

import numpy as np

class GAN():
    def __init__(self):
        self.channels = 1
        self.latent_dim = 100
        rescale_factor = 32

        optimizer = Adam(0.0001, 0.5)

        self.logdir = "./logs"

        # Empty any old log directory
        for the_file in os.listdir(self.logdir):
            file_path = os.path.join(self.logdir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

        # Empty the generated image directory
        for the_file in os.listdir("./images"):
            file_path = os.path.join("./images", the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)


        # Load the dataset
        filelist = glob.glob("./source_imgs/*.jpg")
        imgs = [Image.open(fname) for fname in filelist]


        self.target_size  = (max([x.size[0] for x in imgs]),
                             max([x.size[1] for x in imgs]))

        self.target_size = tuple([x//rescale_factor for x in self.target_size])

        self.X_train = []

        for img in imgs:
            old_size = img.size
            ratio = min(self.target_size[0]/old_size[0],
                        self.target_size[1]/old_size[1])

            new_size = tuple([int(x*ratio) for x in old_size])
            img = img.resize(new_size, Image.ANTIALIAS)
            img = PIL.ImageOps.invert(img)
            new_img = Image.new("L", self.target_size)
            new_img.paste(img, ((self.target_size[0]-new_size[0])//2,
                                (self.target_size[1]-new_size[1])//2))
            self.X_train.append(new_img)

        self.X_train = np.stack(self.X_train)

        self.img_shape = (self.target_size[1],
                          self.target_size[0],
                          self.channels)

        # Rescale -1 to 1
        self.X_train = self.X_train / 127.5 - 1.
        self.X_train = np.expand_dims(self.X_train, axis=3)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()


        model.add(Conv2D(16, (5, 5),
                         activation='relu',
                         padding='same',
                         input_shape=self.img_shape))

        model.add(Conv2D(8, (7, 7),
                         activation='relu',
                         padding='same',
                         input_shape=self.img_shape))

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        tensorboard = TensorBoard(log_dir=self.logdir)
        tensorboard.set_model(self.discriminator)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            imgs = self.X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            if epoch == 0 or accuracy < 80:
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                # Test the discriminator
                d_loss_real = self.discriminator.test_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.test_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            accuracy = 100*d_loss[1]

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            if epoch == 0 or accuracy > 20:
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)
            else:
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.test_on_batch(noise, valid)

            tensorboard.on_epoch_end(epoch, {'generator loss': g_loss, 'discriminator loss': d_loss[0], 'Accuracy': accuracy})

            # Plot the progress
            print(f"{epoch} [D loss: {d_loss[0]}, " +
                  f"acc.: {accuracy}%] [G loss: {g_loss}]")

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        tensorboard.on_train_end()

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images from [-1, 1] to [1, 0] (invert)
        gen_imgs = -0.5 * gen_imgs - 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=16, sample_interval=20)
