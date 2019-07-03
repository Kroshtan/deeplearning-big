import numpy as np
import numpy.matlib as matlib
import cv2
from os import listdir, mkdir
from os.path import isfile, isdir, join
from copy import deepcopy
from tqdm import tqdm

ROBINPATH = '../ROBIN'
COMPLEXPATH = '../Dataset_complex'
OUTPATH = 'augmented'
RESIZE_FACTOR = 32
BATCH_SIZE = 1


def padd_h(image, padding_height):
    m = np.array([255], dtype=np.uint8)
    (_, maxx) = image.shape
    padding_top = matlib.repmat(m, padding_height // 2, maxx)
    padding_bot = matlib.repmat(m,
                                padding_height // 2 + padding_height % 2,
                                maxx)  # take care of the floored pixel

    return np.concatenate((padding_top, image, padding_bot), axis=0)


def padd_v(image, padding_width):
    m = np.array([255], dtype=np.uint8)
    (maxy, _) = image.shape
    padding_left = matlib.repmat(m, maxy, padding_width // 2)
    padding_right = matlib.repmat(m,
                                  maxy,
                                  padding_width // 2 + padding_width % 2)

    return np.concatenate((padding_left, image, padding_right), axis=1)


def padd_to_biggest_file(image, img_size):
    (maxy, maxx) = image.shape
    (maxh, maxw) = img_size
    img = padd_h(image, maxh - maxy)
    return padd_v(img, maxw - maxx)


def scale_and_padd_to_biggest_file(image, img_size):
    # todo implement scaling
    (maxy, maxx) = image.shape
    (maxh, maxw) = img_size
    # take smallest scale to not go out of bounds
    scale = maxw / maxx if maxh / maxy > maxw / maxx else maxh / maxy

    image = cv2.resize(image, (int(maxx * scale), int(maxy * scale)))

    return padd_to_biggest_file(image, img_size)


def loadAllFiles(path):
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    folders = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
    for folder in folders:
        temp = [join(folder, f) for f in listdir(folder)
                if isfile(join(folder, f))]
        for f in temp:
            files.append(f)  # slow and ugly, fix later

    print("found %d files on %s" % (len(files), str(path)))
    return files


def rotate_image(img):
    (maxy, maxx) = np.shape(img)
    rotated = np.zeros((maxx, maxy), dtype=np.uint8)

    for y in range(maxy):
        for x in range(maxx):
            rotated[x, y] = img[y, x]

    return rotated


def augment_images(images, outpath, filename, img_size, flip=True,
                   saveiter=1000000):
    '''
    Flip should be set to false for the advanced dataset
    saveiter is the iteration at which the images get saved (and the ram freed)

    Rotation code is now very ugly, but this is due to the rotation code being
    very inefficient, so I tried to use it as little as possible
    '''
    final_images = []
    saveidx = 0

    for idx, imgpath in tqdm(enumerate(images)):
        original_img = cv2.imread(imgpath)

        # set to one channel
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        original_img = scale_and_padd_to_biggest_file(original_img, img_size)
        original_img = cv2.resize(original_img, img_size)

        final_images.append(deepcopy(original_img))

        # add rotations
        img = deepcopy(original_img)
        final_images.append(np.flip(img, 0))
        rot_img = rotate_image(img)
        rot_img = scale_and_padd_to_biggest_file(rot_img, img_size)
        rot_img = cv2.resize(rot_img, img_size)

        final_images.append(rot_img)
        final_images.append(np.flip(rot_img, 1))

        # mirror and rotate
        if flip:
            img = np.flip(original_img, 1)
            final_images.append(deepcopy(img))
            final_images.append(np.flip(img, 0))
            rot_img = rotate_image(img)

            rot_img = scale_and_padd_to_biggest_file(rot_img, img_size)
            rot_img = cv2.resize(rot_img, img_size)
            final_images.append(rot_img)
            final_images.append(np.flip(rot_img, 1))

        # save images to numpy array
        if idx % saveiter == 0:
            np.save(join(outpath, '%s_%d' % (filename, saveidx)), final_images)
            saveidx += 1
            final_images = []

    # for i in final_images:
    #     print(i.shape)

    # np.save(join(outpath, '%s_%d' % (filename, saveidx)), final_images)

    # print("augmented and saved all files")


# def get_max_dims(images, resize_factor):
#     # Get size of largest image
#     max_height = 0
#     max_width = 0

#     for idx, imgpath in enumerate(images):
#         original_img = cv2.imread(imgpath)
#         (height, width, _) = original_img.shape
#         if height > max_height:
#             max_height = height
#         if width > max_width:
#             max_width = width
#     resize_shape_height = max_height // resize_factor
#     resize_shape_width = max_width // resize_factor
#     while resize_shape_height % 32 != 0:
#         resize_shape_height += 1
#     while resize_shape_width % 32 != 0:
#         resize_shape_width += 1

#     return (resize_shape_height, resize_shape_width, max_height, max_width)
