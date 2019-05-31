import numpy as np
import cv2
from os import listdir
from os.path import isfile, isdir, join, abspath

PATH = abspath("../ROBIN")
OUTPATH = abspath("../")

#DONE 	load in images
#		aug to 8 different images (rotate, vertical mirror and rotate)
#		save images to numpy array? npz?

def loadAllFiles():
    files =  [join(PATH, f) for f in listdir(PATH) if isfile(join(PATH, f))]
    folders = [join(PATH, f) for f in listdir(PATH) if isdir(join(PATH, f))]
    for folder in folders:
    	temp =  [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    	for f in temp:
    		files.append(f) #slow and ugly, fix later

    print("found %d files:" % (len(files)))
    return files

def augment_images(images):
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
			rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), rot, .5)
			img = cv2.warpAffine(original_img, rotationMatrix, (width, height))
			final_images.append(img)

		#mirror and rotate
		original_img = np.flip(original_img, 0);
		final_images.append(original_img)
		for rot in range(90, 360, 90):
			rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), rot, .5)
			img = cv2.warpAffine(original_img, rotationMatrix, (width, height))
			final_images.append(img)

		if idx % 100 == 0:
			np.save(join(OUTPATH, 'augmented_data_%d' % (saveidx)), final_images)
			# print("saved 100 images to ", join(OUTPATH, 'augmented_data_%d' % (saveidx)))
			saveidx += 1
			final_images = []

	np.save(join(OUTPATH, 'augmented_data_%d' % (saveidx)), final_images)
	# print("saved final images to ", join(OUTPATH, 'augmented_data_%d' % (saveidx)))
	# print("finished augmenting images")
	print("augmented and saved all files")


if __name__ == '__main__':
	files = loadAllFiles()
	augment_images(files)	
	print("FINIHED")
