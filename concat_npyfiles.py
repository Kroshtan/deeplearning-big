import numpy as np
import glob
SAVE_FILE_ROBIN = "robin_train.npy"
in_path = 'augmented/'
SAVE_FILE_COMPLEX = 'complex_train.npy'

numpy_vars = []
for np_name in glob.glob(in_path+'robin_data*'):
	temp = np.load(np_name, allow_pickle=True)
	numpy_vars.append(temp)
flat_arr = [item for sublist in numpy_vars for item in sublist]

arr_to_save = np.asarray(flat_arr)
print(arr_to_save.shape)
np.save(SAVE_FILE_COMPLEX, arr_to_save)


numpy_vars = []
for np_name in glob.glob(in_path+'complex_data*'):
	temp = np.load(np_name, allow_pickle=True)
	numpy_vars.append(temp)
flat_arr = [item for sublist in numpy_vars for item in sublist]

arr_to_save = np.asarray(flat_arr)
print(arr_to_save.shape)
np.save(SAVE_FILE_ROBIN, arr_to_save)