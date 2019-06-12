import numpy as np
import numpy.matlib as matlib
import cv2
from os import listdir
from os.path import isfile, isdir, join, abspath
from copy import deepcopy

ROBINPATH 	= abspath("../ROBIN")
COMPLEXPATH = abspath("../Dataset_complex")
OUTPATH 	= abspath("../")
RESIZE_FACTOR = 32

def padd_h(image, padding_height):
	m = np.array([255], dtype=np.uint8)
	(_, maxx) = image.shape
	padding_top = matlib.repmat(m, padding_height // 2, maxx)
	padding_bot = matlib.repmat(m, padding_height // 2 + padding_height % 2, maxx) #take care of the floored pixel

	return np.concatenate((padding_top, image, padding_bot), axis=0)

def padd_v(image, padding_width):
	m = np.array([255], dtype=np.uint8)
	(maxy, _) = image.shape
	padding_left = matlib.repmat(m, maxy, padding_width // 2)
	padding_right = matlib.repmat(m, maxy, padding_width // 2 + padding_width % 2)

	return np.concatenate((padding_left, image, padding_right), axis=1)

def padd_to_biggest_file(image, maxh, maxw):
	(maxy, maxx) = image.shape
	img = padd_h(image, maxh - maxy)
	return padd_v(img, maxw - maxx)

def scale_and_padd_to_biggest_file(image, maxh, maxw):
	## todo implement scaling
	(maxy, maxx) = image.shape
	scale = maxw / maxx if maxh / maxy > maxw / maxx else maxh / maxy #take smallest scale to not go out of bounds

	image = cv2.resize(image, (int(maxx * scale),int(maxy * scale) ) )

	return padd_to_biggest_file(image, maxh, maxw)


def loadAllFiles(path):
    files =  [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    folders = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
    for folder in folders:
    	temp =  [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    	for f in temp:
    		files.append(f) #slow and ugly, fix later

    print("found %d files on %s" % (len(files), str(path)))
    return files

def rotate_image(img):
	(maxy, maxx) = np.shape(img)
	rotated = np.zeros((maxx, maxy), dtype=np.uint8)

	for y in range(maxy):
		for x in range(maxx):
			rotated[x, y] = img[y, x]

	return rotated



def augment_images(images, filename, flip = True, saveiter = 1000000, saveaspng=False, resize_height=0, resize_width=0, max_height = 0, max_width = 0):
	'''
	Flip should be set to false for the advanced dataset
	saveiter is the iteration at which the images get saved (and the ram freed)

	Rotation code is now very ugly, but this is due to the rotation code being very inefficient, so I tried to use 
	it as little as possible
	'''
	final_images = []
	saveidx = 0

	for idx, imgpath in enumerate(images):
		original_img = cv2.imread(imgpath)
		original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) #set to one channel
		original_img = scale_and_padd_to_biggest_file(original_img, max_height, max_width)
		original_img = cv2.resize(original_img, (width,height))
		#append original

		cv2.imshow("", original_img)
		cv2.waitKey(0)

		final_images.append(deepcopy(original_img))

		#add rotations
		img = deepcopy(original_img)
		final_images.append(np.flip(img, 0))
		rot_img = rotate_image(img)
		final_images.append(rot_img)
		final_images.append(np.flip(rot_img, 1))

		
		

		#mirror and rotate
		if flip:
			img = np.flip(original_img, 1);
			final_images.append(deepcopy(img))
			final_images.append(np.flip(img, 0))
			rot_img = rotate_image(img)
			final_images.append(rot_img)
			final_images.append(np.flip(rot_img, 1))

		#save images to numpy array
		if idx % saveiter == 0 and not saveaspng and False: #Don't run this part, we want to save in 1 file 
			np.save(join(OUTPATH, '%s_%d' % (filename, saveidx)), final_images)
			saveidx += 1
			final_images = []

		#save all images individually
		if saveaspng:
			for img in final_images:
				cv2.imwrite('filename_%d.png' % (saveidx), img)
				saveidx += 1
			final_images = []



	final_images = np.array(final_images)
	np.save(join(OUTPATH, '%s_%d' % (filename, saveidx)), final_images)
	print("augmented and saved all files")

def find_dimensions(images):
	# Get size of largest image
	max_height = 0
	max_width = 0

	for idx, imgpath in enumerate(images):
		original_img = cv2.imread(imgpath)
		(height, width, _) = original_img.shape
		if height > max_height:
			max_height = height
		if width > max_width:
			max_width = width
	resize_shape_height = max_height//RESIZE_FACTOR
	resize_shape_width = max_width//RESIZE_FACTOR
	return (resize_shape_height, resize_shape_width, max_height, max_width)

def get_max_img_sizes(images1, images2):
	(height1, width1, mh1, mw1) = find_dimensions(images1)
	(height2, width2, mh2, mw2) = find_dimensions(images2)
	# print(height1, width1)
	# print(height2, width2)

	output_height = height1 if height1 > height2 else height2 
	output_width = width1 if width1 > width2 else width2

	max_height = mh1 if mh1 > mh2 else mh2
	max_width = mw1 if mw1 > mw2 else mw2

	return (output_height, output_width, max_height, max_width)

if __name__ == '__main__':
	files1 = loadAllFiles(ROBINPATH)
	files2 = loadAllFiles(COMPLEXPATH)

	(height, width, max_height, max_width) = get_max_img_sizes(files1, files2)
	print(height, width)

	augment_images(files1, 'robin_data', resize_height=height, resize_width=width, max_height = max_height, max_width=max_width)
	augment_images(files2, 'complex_data', resize_height=height, resize_width=width, max_height = max_height, max_width=max_width)

	print("FINIHED")