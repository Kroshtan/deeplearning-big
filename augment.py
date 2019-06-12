import numpy as np
import cv2
from os import listdir
from os.path import isfile, isdir, join, abspath

ROBINPATH 	= abspath("../ROBIN")
COMPLEXPATH = abspath("../Dataset_complex")
OUTPATH 	= abspath("../")


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
	rotated = np.zeros((maxx, maxy), dtype=np.uint)

	for y in range(maxy):
		for x in range(maxx):
			rotated[x, y] = img[y, x]

	return rotated



def augment_images(images, filename, flip = True, saveiter = 100, saveaspng=False):
	'''
	Flip should be set to false for the advanced dataset
	saveiter is the iteration at which the images get saved (and the ram freed)
	'''
	final_images = []
	saveidx = 0
	for idx, imgpath in enumerate(images):
		original_img = cv2.imread(imgpath)
		original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) #set to one channel

		#append original
		final_images.append(original_img)

		#rotate
		(height, width) = np.shape(original_img)
		for rot in range(90, 360, 90):
			# rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), rot, .5)
			# img = cv2.warpAffine(original_img, rotationMatrix, (width, height))
			img = rotate_image(original_img)
			final_images.append(img)

		#mirror and rotate
		if flip:
			original_img = np.flip(original_img, 0);
			final_images.append(original_img)
			for rot in range(90, 360, 90):
				# rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), rot, .5)
				# img = cv2.warpAffine(original_img, rotationMatrix, (width, height))
				img = rotate_image(original_img)
				final_images.append(img)

		#save images to numpy array
		if idx % saveiter == 0 and not saveaspng:
			np.save(join(OUTPATH, '%s_%d' % (filename, saveidx)), final_images)
			saveidx += 1
			final_images = []

		#save all images individually
		if saveaspng:
			for img in final_images:
				cv2.imwrite('filename_%d.png' % (saveidx), img)
				saveidx += 1
			final_images = []




	np.save(join(OUTPATH, '%s_%d' % (filename, saveidx)), final_images)
	print("augmented and saved all files")


if __name__ == '__main__':
	files = loadAllFiles(ROBINPATH)
	augment_images(files, 'robin_data')

	files = loadAllFiles(COMPLEXPATH)
	augment_images(files, 'complex_data', False, 50) #the complex images are bigger, and should be saved less frequently for RAM

	print("FINIHED")