from __future__ import print_function, division
import augment as aug
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D, LeakyReLU, Conv2DTranspose, Dropout
from keras.layers import concatenate, UpSampling2D, AveragePooling2D
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

# RESIZE_FACTOR = 24
IMG_SIZE = (128, 128, 1)
LATENT_SIZE = (16, 16)
ratio = (IMG_SIZE[0]//LATENT_SIZE[0], IMG_SIZE[1]//LATENT_SIZE[1])
TRAIN_ON_ROBIN = True
TRAIN_ON_COMPLEX = False

NPY_SAVEFILE = 'traindata.npy'
IMAGE_DIR = 'images/'
LOG_DIR = './logs'

EPOCHS = 1000000
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
DECAY = 1e-8
P_FLIP_LABEL = 0.05
LRELU_FACTOR = 0.2
ADD_LABEL_NOISE = False  # Tends to set accuracy to 0, preventing training. Why??
LABEL_NOISE = 0.01
DROPOUT_RATE = 0.1
SAMPLE_INTERVAL = 20


class GAN():
    def __init__(self):
        self.img_per_npy = os.path.join(OUTPATH, os.listdir(OUTPATH)[0])
        self.img_per_npy = np.load(self.img_per_npy, allow_pickle=True)
        self.img_per_npy = self.img_per_npy.shape[0]

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
        z = Input(shape=LATENT_SIZE)
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

        inp = Input(shape=LATENT_SIZE)
        inp_resh = Reshape(LATENT_SIZE + (1,))(inp)

        n_filt = [128, 64, 32, 8]
        kernel_size = [9, 9, 9, 9]
        pool_size = [ratio, 4, 2, 2]

        numlayers = len(n_filt)

        g = inp_resh
        for block in range(numlayers):

            g = Conv2D(filters=n_filt[block],
                       kernel_size=kernel_size[block],
                       padding="same",
                       strides=(1 if not block else pool_size[block]))(g)
            # g = Dropout(rate=DROPOUT_RATE)(g)

            g = Conv2DTranspose(filters=1,
                                kernel_size=kernel_size[block],
                                padding="same",
                                strides=pool_size[block])(g)
            # g = Dropout(rate=DROPOUT_RATE)(g)

        # g = Conv2DTranspose(filters=1,
        #                     kernel_size=1,
        #                     padding="valid")(g)

        out = Activation('tanh')(g)

        model = Model(inputs=inp, outputs=out)

        model.summary()

        return model

    def build_discriminator(self):

        d_in = Input(shape=IMG_SIZE)

        d = Conv2D(filters=8,
                   kernel_size=4,
                   padding='valid',
                   strides=1)(d_in)

        r1 = MaxPooling2D(pool_size=3)(d)

        d = Conv2D(filters=16,
                   kernel_size=8,
                   padding='valid',
                   strides=2)(d)

        r2 = MaxPooling2D(pool_size=3)(d)

        d = Conv2D(filters=16,
                   kernel_size=8,
                   padding='valid',
                   strides=2)(d)
        r3 = MaxPooling2D(pool_size=3)(d)

        d = Conv2D(filters=48,
                   kernel_size=8,
                   padding='valid',
                   strides=2)(d)

        r4 = MaxPooling2D(pool_size=3)(d)

        d = Conv2D(filters=128,
                   kernel_size=8,
                   padding='valid',
                   strides=2)(d)

        d = Flatten()(d)
        d = concatenate([d,
                         Flatten()(r1),
                         Flatten()(r2),
                         Flatten()(r3),
                         Flatten()(r4)])

        d = Dense(units=196)(d)
        d = LeakyReLU(alpha=LRELU_FACTOR)(d)
        d = Dropout(rate=DROPOUT_RATE)(d)
        d = Dense(units=128)(d)
        d = LeakyReLU(alpha=LRELU_FACTOR)(d)
        d = Dropout(rate=DROPOUT_RATE)(d)

        d = Dense(units=1, activation='sigmoid')(d)

        d = Model(inputs=d_in, outputs=d)
        d.summary()

        return d

    def train(self, epochs, batch_size=BATCH_SIZE, sample_interval=50):

        tensorboard = TensorBoard(log_dir=LOG_DIR)
        tensorboard.set_model(self.discriminator)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Detect batch size in npys

            batch_size = min(self.img_per_npy, batch_size)

            idx = np.random.randint(0, self.img_per_npy-1, batch_size)

            # Select a random batch of images
            self.X_train = os.path.join(OUTPATH,
                                        np.random.choice(os.listdir(OUTPATH)))
            self.X_train = np.load(self.X_train, allow_pickle=True)
            self.X_train = self.X_train[idx]
            self.X_train = np.expand_dims(self.X_train, axis=3)
            self.X_train = self.X_train / (-255/2) + 1

            noise = np.random.normal(-1, 1, ((batch_size,) + LATENT_SIZE))

            # Adversarial ground truths

            valid = np.ones((batch_size,))
            if ADD_LABEL_NOISE:
                valid -= np.random.uniform(high=LABEL_NOISE,
                                           size=(batch_size,))
            for img in range(batch_size):
                if np.random.rand() < P_FLIP_LABEL:
                    valid[img] = 1 - valid[img]

            fake = np.zeros((batch_size,))
            if ADD_LABEL_NOISE:
                fake += np.random.uniform(high=LABEL_NOISE,
                                          size=(batch_size,))
                print(fake)
            for img in range(batch_size):
                if np.random.rand() < P_FLIP_LABEL:
                    fake[img] = 1 - fake[img]

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
            noise = np.random.normal(-1, 1, ((batch_size,) + LATENT_SIZE))

            if epoch == 0 or accuracy > 67:
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
                self.sample_images(epoch, accuracy, real_imgs=self.X_train)

        tensorboard.on_train_end(tensorboard)
        self.discriminator.save('discriminator.h5')
        self.generator.save('generator.h5')

    def sample_images(self, epoch, accuracy, real_imgs):
        r = 2
        noise = np.random.normal(-1, 1, ((r-1,) + LATENT_SIZE))
        gen_imgs = self.generator.predict(noise)

        # Select a random image
        real_imgs = real_imgs[0, :, :, 0]

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

        if TRAIN_ON_COMPLEX:
            files_complex = aug.loadAllFiles(COMPLEXPATH)


        if TRAIN_ON_ROBIN:
            print(f"Preparing the ROBIN files...")
            aug.augment_images(images=files_robin,
                               outpath=OUTPATH,
                               filename='robin_data',
                               img_size=IMG_SIZE[:2],
                               saveiter=BATCH_SIZE)

        if TRAIN_ON_COMPLEX:
            print(f"Preparing the COMPLEX files...")
            aug.augment_images(images=files_complex,
                               outpath=OUTPATH,
                               filename='complex_data',
                               img_size=IMG_SIZE[:2],
                               saveiter=BATCH_SIZE)

    # Train the GAN
    gan = GAN()
    gan.train(epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL)
