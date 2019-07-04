# Deep-learning project:
GAN for building floorplans

## Run instructions
1. Download the data:

<br />
Simple ROBIN dataset: https://github.com/gesstalt/ROBIN/blob/master/ROBIN.zip
<br />
Complex dataset: http://dag.cvc.uab.es/resources/floorplans/

2. Set the flag USE_PRETRAINED_WEIGHTS = False, to train on the new images. Otherwise you need to provide the program with pretrained weights.

3. Run the GAN with ```python3 new_gan.py```.  


### Augmentation techniques
1. Flip 
2. Rotate 90* (4x)
So in total 8 augmented images (including original image) are created per input image. 
