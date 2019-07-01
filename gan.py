from __future__ import print_function, division
import augment as aug
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D, LeakyReLU, Conv2DTranspose, Dropout
from keras.models import Model, Sequential
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

EPOCHS = 1000000
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
DECAY = 1e-9
P_FLIP_LABEL = 0.05
LRELU_FACTOR = 0.1
LABEL_NOISE = (0, 0)
DROPOUT_RATE = 0.1
SAMPLE_INTERVAL = 50


class GAN():
    def __init__(self):
        self.latent_dim = (5, 5)
        random_file = os.path.join(OUTPATH, os.listdir(OUTPATH)[0])
        self.img_size = np.load(random_file, allow_pickle=True)[0].shape
        self.img_size += (1,)  # add color channel for conv layers

        self.this_npy_num_imgs = os.path.join(OUTPATH, os.listdir(OUTPATH)[0])
        self.this_npy_num_imgs = np.load(self.this_npy_num_imgs,
                                         allow_pickle=True)
        self.this_npy_num_imgs = self.this_npy_num_imgs.shape[0]

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

        model = Sequential()

        n_filt = 12

        model.add(Reshape((np.prod(self.latent_dim),), input_shape=self.latent_dim))

        model.add(Dense(units=128))
        model.add(Dense(units=np.prod(self.latent_dim)))

        model.add(Reshape((self.latent_dim + (1,))))

        model.add(Conv2D(filters=n_filt,
                         kernel_size=5,
                         padding="same"))

        model.add(Conv2DTranspose(filters=n_filt,
                                  kernel_size=(3, 7),
                                  strides=3,
                                  padding='valid'))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(LeakyReLU(alpha=LRELU_FACTOR))

        model.add(Conv2DTranspose(filters=1,
                                  kernel_size=(5, 6),
                                  strides=5,
                                  padding='valid'))

        model.add(Activation('tanh'))

        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(filters=8,
                         kernel_size=3,
                         padding='same',
                         input_shape=self.img_size))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Flatten())
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(LeakyReLU(alpha=LRELU_FACTOR))

        model.add(Dense(units=64))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(LeakyReLU(alpha=LRELU_FACTOR))

        model.add(Dense(units=1))

        model.summary()

        return model

    def train(self, epochs, batch_size=BATCH_SIZE, sample_interval=50):

        tensorboard = TensorBoard(log_dir=LOG_DIR)
        tensorboard.set_model(self.discriminator)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Detect batch size in npys

            batch_size = min(self.this_npy_num_imgs, batch_size)

            idx = np.random.randint(0, self.this_npy_num_imgs-1, batch_size)

            # Select a random batch of images
            self.X_train = os.path.join(OUTPATH,
                                        np.random.choice(os.listdir(OUTPATH)))
            self.X_train = np.load(self.X_train, allow_pickle=True)
            self.X_train = self.X_train[idx]
            self.X_train = np.expand_dims(self.X_train, axis=3)
            self.X_train = self.X_train / (-255/2) + 1

            noise = np.random.normal(-1, 1, ((batch_size,) + self.latent_dim))

            # Adversarial ground truths
            if np.random.rand() < P_FLIP_LABEL:
                valid = np.zeros((BATCH_SIZE,)) + np.random.uniform(
                    high=LABEL_NOISE[1],
                    size=(BATCH_SIZE,))
                fake = np.ones((BATCH_SIZE,)) - np.random.uniform(
                    high=LABEL_NOISE[1],
                    size=(BATCH_SIZE,))
            else:
                valid = np.ones((BATCH_SIZE,)) - np.random.uniform(
                    high=LABEL_NOISE[1],
                    size=(BATCH_SIZE,))
                fake = np.zeros((BATCH_SIZE,)) + np.random.uniform(
                    high=LABEL_NOISE[1],
                    size=(BATCH_SIZE,))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            if epoch == 0 or accuracy < 80:
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(self.X_train,
                                                                valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
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
                                             'Accuracy': accuracy,
                                             'Comb. loss': g_loss + d_loss[0]})

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print(f"@ {epoch:{len(str(EPOCHS))}}:\t"
                      f"Accuracy: {int(accuracy):3}%\t"
                      f"G-Loss: {g_loss:6.3f}\t"
                      f"D-Loss: {d_loss[0]:6.3f}\t"
                      f"Combined: {g_loss+d_loss[0]:6.3f}")
                self.sample_images(epoch, accuracy)

        tensorboard.on_train_end(tensorboard)
        self.discriminator.save('discriminator.h5')
        self.generator.save('generator.h5')

    def sample_images(self, epoch, accuracy):
        r = 2
        noise = np.random.normal(-1, 1, ((r-1,) + self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Select a random image
        real_imgs = os.path.join(OUTPATH, np.random.choice(os.listdir(OUTPATH)))
        real_imgs = np.load(real_imgs, allow_pickle=True)
        real_imgs = real_imgs[np.random.randint(0, BATCH_SIZE*8), :, :]
        real_imgs = real_imgs / (-255/1)

        fig, axs = plt.subplots(r)
        fig.suptitle(f"Epoch {epoch}\nAccuracy {int(accuracy)}%")


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
                               max_width=max_width,
                               saveiter=BATCH_SIZE)

        if TRAIN_ON_COMPLEX:
            print(f"Preparing the COMPLEX files...")
            aug.augment_images(images=files_complex,
                               outpath=OUTPATH,
                               filename='complex_data',
                               resize_height=height,
                               resize_width=width,
                               max_height=max_height,
                               max_width=max_width,
                               saveiter=BATCH_SIZE)

    # Train the GAN
    gan = GAN()
    gan.train(epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL)
