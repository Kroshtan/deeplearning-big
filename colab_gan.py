#@title  { form-width: "250px" }
from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Conv2DTranspose, MaxPooling2D, Concatenate, LeakyReLU
from keras.layers import Dropout
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.callbacks import TensorBoard
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
import os
import shutil
from random import choice, randint
import glob, os
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
import sys
import numpy as np

PREPARE_COLAB_DATA = False
RUN_ON_COLAB = True
NPY_SAVEFILE = 'traindata.npy'
IMAGE_DIR = 'images/'
LOG_DIR = './logs'
TRAIN_ON_AUGMENTED = True
SIMPLE_DATA = ['./robin_data_0.npy']
COMPLEX_DATA = ['./complex_data_0.npy']
#ALL_NPYS = os.listdir('./augmented')

EPOCHS = 30000
BATCH_SIZE = 16
SAMPLE_INTERVAL = 100
RESCALE_FACTOR = 32
# TRAIN_ON_COMPLEX = False

class GAN():
    def __init__(self):
        self.channels = 1
        self.latent_dim = 50

        #self.img_size = np.load('./augmented/'+ALL_NPYS[0], allow_pickle=True)[0].shape
        self.img_size = np.load(SIMPLE_DATA[0])[0].shape
        self.img_size += (1,)  # add color channel for conv layers

        optimizer = Adam(1e-3, decay=1e-4)

        # Empty any old image directory
        if os.path.exists(IMAGE_DIR):
            shutil.rmtree(IMAGE_DIR)
            print("Removed old image directory.")

        os.mkdir(IMAGE_DIR)
        print('Created new image directory.')

        if not RUN_ON_COLAB:
            # Empty any old log directory
            if os.path.exists(LOG_DIR):
                shutil.rmtree(LOG_DIR)
                print("Removed old log directory.")

            os.mkdir(LOG_DIR)
            print('Created new log directory.')

        try:
            # Empty the generated image directory
            for the_file in os.listdir("./images"):
                file_path = os.path.join("./images", the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        except:
            print("./images dir does not yet exist")

        # Load the dataset
        filelist = glob.glob("./source_imgs/*.jpg")
        imgs = [Image.open(fname) for fname in filelist]
        # if RUN_ON_COLAB:
        #     try:
        #         os.mkdir(IMAGE_DIR)
        #         print("Created output images directory...")
        #     except:
        #         print("Output images directory already exists!")
        #     if TRAIN_ON_AUGMENTED:
        #         self.X_train = np.load(COMPLEX_DATA[0], allow_pickle=True)
        #     else:
        #         self.X_train = np.stack(np.load(NPY_SAVEFILE, allow_pickle=True))
        #     self.X_train = np.expand_dims(self.X_train, axis=3)
        #     target_size = (max([x.shape[1] for x in self.X_train]), max([x.shape[0] for x in self.X_train]))
        #     #target_size = (self.X_train[0].shape[1], self.X_train[1].shape[0])
        #     self.img_shape = (target_size[1], target_size[0], self.channels)

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

        inp = Input(shape=(self.latent_dim,))

        layer1 = Dense(256,
                       input_shape=(self.latent_dim,))(inp)
        layer1 = LeakyReLU()(layer1)
        layer1 = BatchNormalization(momentum=0.8)(layer1)
        # layer1 = Dropout(rate=0.5)(layer1)

        layer2 = Dense(256)(layer1)
        layer2 = LeakyReLU()(layer2)
        layer2 = BatchNormalization(momentum=0.8)(layer2)

        # layer2 = Dropout(rate=0.5)(layer2)

        # layer3 = Dense(128)(layer2)
        # layer3 = LeakyReLU()(layer3)
        # layer3 = BatchNormalization(momentum=0.8)(layer3)
        # layer3 = Dropout(rate=0.5)(layer3)

        concat = Concatenate(axis=-1)([layer1, layer2])

        pre_out = Dense(np.prod(self.img_size), activation='tanh')(concat)

        out = Reshape(target_shape=self.img_size)(pre_out)

        model = Model(inputs=inp, outputs=out)

        model.summary()

        return model

    def build_discriminator(self):

        inp = Input(shape=self.img_size)

        conv1 = Conv2D(filters=4,
                       kernel_size=(3, 3),
                       activation='relu',
                       padding='same')(inp)
        conv1 = MaxPooling2D(pool_size=(4, 4))(conv1)
        flat_conv1 = Flatten()(conv1)
        flat_conv1 = Dense(56, activation='relu')(flat_conv1)

        conv2 = Conv2D(filters=4,
                       kernel_size=(5, 5),
                       activation='relu',
                       padding='same')(conv1)
        # conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        flat_conv2 = Flatten()(conv2)
        flat_conv2 = Dense(56, activation='relu')(flat_conv2)


        fc = Concatenate()([flat_conv1, flat_conv2])

        fc = Dense(128, activation='relu')(fc)

        out = Dense(1, activation='sigmoid')(fc)

        model = Model(inputs=inp, outputs=out)
        model.summary()

        return model

    def train(self, epochs, batch_size=BATCH_SIZE, sample_interval=50):

        tensorboard = TensorBoard(log_dir=LOG_DIR)
        tensorboard.set_model(self.discriminator)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            #this_npy = randint(0, len(ALL_NPYS)-1)
            #this_npy_num_imgs = np.load('./augmented/'+ALL_NPYS[this_npy], allow_pickle=True).shape[0]
            this_npy_num_imgs = np.load(SIMPLE_DATA[0], allow_pickle=True).shape[0]
            batch_size = min(this_npy_num_imgs, batch_size)


            # Adversarial ground truths
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))


            idx = np.random.randint(0, this_npy_num_imgs-1, batch_size)


            #self.X_train = np.load('./augmented/'+ALL_NPYS[this_npy], allow_pickle=True)[idx]
            self.X_train = np.load(SIMPLE_DATA[0], allow_pickle=True)[idx]
            self.X_train = np.expand_dims(self.X_train, axis=3)
            self.X_train = self.X_train / (255/2) - 1

            noise = np.random.normal(-1, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            if epoch == 0 or accuracy < 80:
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(self.X_train, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                # Test the discriminator
                d_loss_real = self.discriminator.test_on_batch(self.X_train, valid)
                d_loss_fake = self.discriminator.test_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            accuracy = 100*d_loss[1]

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(-1, 1, (batch_size, self.latent_dim))

            if epoch == 0 or accuracy > 20:
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)
            else:
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.test_on_batch(noise, valid)

            tensorboard.on_epoch_end(epoch, {'generator loss': g_loss, 'discriminator loss': d_loss[0], 'Accuracy': accuracy})

            # Plot the progress
            if RUN_ON_COLAB:
                if (epoch % 10) == 0:
                    print(f"{epoch} [D loss: {d_loss[0]}, " +
                  f"acc.: {accuracy}%] [G loss: {g_loss}]")
            else:
                print(f"{epoch} [D loss: {d_loss[0]:.3f}, " +
                  f"acc.: {accuracy:.2f}%] [G loss: {g_loss:.3f}]")

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        tensorboard.on_train_end()
        self.discriminator.save('discriminator.h5')
        self.generator.save('generator.h5')

    def sample_images(self, epoch):
        r = 3
        noise = np.random.normal(-1, 1, (r, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        # Rescale images from [-1, 1] to [1, 0] (invert)
        real_imgs = self.X_train[np.random.choice(self.X_train.shape[0]), :, :, 0]
        # gen_imgs = -0.5 * gen_imgs - 0.5
        # real_imgs = -0.5 * real_imgs - 0.5

        fig, axs = plt.subplots(r)

        axs[0].imshow(real_imgs, cmap='gray')
        axs[0].axis('off')

        cnt = 0
        for i in range(r-1):
                axs[i+1].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i+1].axis('off')
                cnt += 1

        fig.savefig(IMAGE_DIR+"%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL)
