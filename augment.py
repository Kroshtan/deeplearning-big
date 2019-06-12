import numpy as np
import cv2
from os import listdir
from os.path import isfile, isdir, join, abspath
from copy import deepcopy

ROBINPATH 	= abspath("../ROBIN")
COMPLEXPATH = abspath("../Dataset_complex")
OUTPATH 	= abspath("../")
RESIZE_FACTOR = 32


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



def augment_images(images, filename, flip = True, saveiter = 1000000, saveaspng=False, height=0, width=0):
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
		original_img = cv2.resize(original_img, (width,height))
		#append original

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
	return (resize_shape_height, resize_shape_width)

def get_max_img_sizes(images1, images2):
	(height1, width1) = find_dimensions(images1)
	(height2, width2) = find_dimensions(images2)
	print(height1, width1)
	print(height2, width2)

	output_height = height1 if height1 > height2 else height2 
	output_width = width1 if width1 > width2 else width2

	return (output_height, output_width)

if __name__ == '__main__':
	files1 = loadAllFiles(ROBINPATH)
	files2 = loadAllFiles(COMPLEXPATH)

	(height, width) = get_max_img_sizes(files1, files2)
	print(height, width)

	augment_images(files1, 'robin_data', height=height, width=width)
	augment_images(files2, 'complex_data', height=height, width=width) #the complex images are bigger, and should be saved less frequently for RAM

	print("FINIHED")