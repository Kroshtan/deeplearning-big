from __future__ import print_function, division
import augment as aug
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D, Concatenate, LeakyReLU, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
from os import mkdir
from os.path import isdir, abspath

ROBINPATH = abspath("./ROBIN")
COMPLEXPATH = abspath("./Dataset_complex")
OUTPATH = abspath("./augmented")

RESIZE_FACTOR = 32
TRAIN_ON_ROBIN = True
TRAIN_ON_COMPLEX = False

NPY_SAVEFILE = 'traindata.npy'
IMAGE_DIR = 'images/'
LOG_DIR = './logs'
SIMPLE_DATA = ['./augmented/robin_data_0.npy']
COMPLEX_DATA = ['./augmented/complex_data_0.npy']

EPOCHS = 30000
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
DECAY = 1e-7
SAMPLE_INTERVAL = 100


class GAN():
    def __init__(self):
        self.channels = 1
        self.latent_dim = (10, 10)

        self.img_size = np.load(SIMPLE_DATA[0])[0].shape
        self.img_size += (1,)  # add color channel for conv layers

        optimizer = Adam(LEARNING_RATE, decay=DECAY)

        # Empty any old log directory
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
            print("Removed old log directory.")

        os.mkdir(LOG_DIR)
        print('Created new log directory.')

        # Empty any old image directory
        if os.path.exists(IMAGE_DIR):
            shutil.rmtree(IMAGE_DIR)
            print("Removed old image directory.")

        os.mkdir(IMAGE_DIR)
        print('Created new image directory.')

        # Empty the generated image directory
        for the_file in os.listdir(IMAGE_DIR):
            file_path = os.path.join(IMAGE_DIR, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=self.latent_dim)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        n_filt = 5

        inp = Input(shape=self.latent_dim)

        layer1 = Flatten()(inp)

        layer1 = Dense(units=n_filt * np.prod(self.img_size))(layer1)

        # layer1 = LeakyReLU()(layer1)

        layer1 = Reshape(target_shape=(self.img_size[0],
                                       self.img_size[1],
                                       n_filt))(layer1)

        # layer1 = Conv2DTranspose(filters=n_filt, kernel_size=8, padding="same")(layer1)
        layer1 = Conv2DTranspose(filters=n_filt, kernel_size=6, padding="same")(layer1)
        layer1 = Conv2DTranspose(filters=1, kernel_size=3, padding="same")(layer1)

        out = Reshape(target_shape=self.img_size)(layer1)
        out = Activation('sigmoid')(layer1)

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
            this_npy_num_imgs = np.load(SIMPLE_DATA[0],
                                        allow_pickle=True)
            this_npy_num_imgs = this_npy_num_imgs.shape[0]
            batch_size = min(this_npy_num_imgs, batch_size)

            # Adversarial ground truths
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            idx = np.random.randint(0, this_npy_num_imgs-1, batch_size)

            self.X_train = np.load(SIMPLE_DATA[0], allow_pickle=True)[idx]
            self.X_train = np.expand_dims(self.X_train, axis=3)
            self.X_train = self.X_train / (255/2) - 1

            noise = np.random.normal(-1, 1, ((batch_size,) + self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            if epoch == 0 or accuracy < 80:
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(self.X_train,
                                                                valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                # Test the discriminator
                d_loss_real = self.discriminator.test_on_batch(self.X_train,
                                                               valid)
                d_loss_fake = self.discriminator.test_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            accuracy = 100*d_loss[1]

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(-1, 1, ((batch_size,) + self.latent_dim))

            if epoch == 0 or accuracy > 20:
                # Train the generator (to have the discriminator label samples
                # as valid)
                g_loss = self.combined.train_on_batch(noise, valid)
            else:
                # Train the generator (to have the discriminator label samples
                # as valid)
                g_loss = self.combined.test_on_batch(noise, valid)

            tensorboard.on_epoch_end(epoch, {'generator loss': g_loss,
                                             'discriminator loss': d_loss[0],
                                             'Accuracy': accuracy})

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print(f"@ {epoch:{len(str(EPOCHS))}}:\t"
                      f"Accuracy: {int(accuracy):3}%\t"
                      f"G-Loss: {g_loss:6.3f}\t"
                      f"D-Loss: {d_loss[0]:6.3f}")
                self.sample_images(epoch)

        tensorboard.on_train_end()
        self.discriminator.save('discriminator.h5')
        self.generator.save('generator.h5')

    def sample_images(self, epoch):
        r = 3
        noise = np.random.normal(-1, 1, ((r,) + self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        # Rescale images from [-1, 1] to [1, 0] (invert)
        real_imgs = self.X_train[np.random.choice(
            self.X_train.shape[0]), :, :, 0]

        fig, axs = plt.subplots(r)

        axs[0].imshow(real_imgs, cmap='gray')
        axs[0].axis('off')

        cnt = 0
        for i in range(r-1):
            axs[i+1].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i+1].axis('off')
            cnt += 1

        fig.savefig(IMAGE_DIR+"%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    # Prepare data if not available
    if not isdir(OUTPATH):
        print(f"Creating new folder {OUTPATH}.")
        mkdir(OUTPATH)
    if not os.listdir(OUTPATH):

        if TRAIN_ON_ROBIN:
            files_robin = aug.loadAllFiles(ROBINPATH)
            (height, width, max_height, max_width) = aug.get_max_dims(
                images=files_robin,
                resize_factor=RESIZE_FACTOR)

        if TRAIN_ON_COMPLEX:
            files_complex = aug.loadAllFiles(COMPLEXPATH)
            (height, width, max_height, max_width) = aug.get_max_dims(
                images=files_complex,
                resize_factor=RESIZE_FACTOR)

        if TRAIN_ON_ROBIN and TRAIN_ON_COMPLEX:
            (height, width, max_height, max_width) = aug.get_max_dims(
                images=files_robin + files_complex,
                resize_factor=RESIZE_FACTOR)

        if TRAIN_ON_ROBIN:
            print(f"Preparing the ROBIN files...")
            aug.augment_images(images=files_robin,
                               outpath=OUTPATH,
                               filename='robin_data',
                               resize_height=height,
                               resize_width=width,
                               max_height=max_height,
                               max_width=max_width)

        if TRAIN_ON_COMPLEX:
            print(f"Preparing the COMPLEX files...")
            aug.augment_images(images=files_complex,
                               outpath=OUTPATH,
                               filename='complex_data',
                               resize_height=height,
                               resize_width=width,
                               max_height=max_height,
                               max_width=max_width)

    # Train the GAN
    gan = GAN()
    gan.train(epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL)
