import numpy as np
import cv2
#FILENAMES = ['complex_data_0.npy', 'complex_data_1.npy', 'complex_data_3.npy', 'complex_data_4.npy']
FILENAMES = ['augmented_data_0.npy']
WIDTH = HEIGHT = 500

def open_file(filename):
	images = np.load(filename, allow_pickle=True)
	for image in images:
		image = cv2.resize(image,(WIDTH,HEIGHT))
		cv2.imshow('image',image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

for file in FILENAMES:
	file = '../'+file
	open_file(file)