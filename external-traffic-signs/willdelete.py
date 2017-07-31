### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob

# size_images = 10
# images = np.zeros((size_images,32,32,3), dtype = np.int32)
# for i in range(size_images):
#    filename = "./external-traffic-signs/p%d_resized.jpg" % i
#    images[i] = cv2.imread(filename)

image_files  = ['external-traffic-signs/' + image_file for image_file in os.listdir('external-traffic-signs')]
images = []
for i, img in enumerate(image_files):
    image = cv2.imread(img)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    images.append(image_gray)
    
images = np.asarray(images)
images = (images - 128)/128 
    
# for image_file in image_files:
#    image = Image.open(image_file)
#    image = image.convert('RGB')
#    image = image.resize((32, 32), Image.ANTIALIAS)
#    image = np.array(list(image.getdata()), dtype='uint8')
#    image = np.reshape(image, (32, 32, 3))
#    images.append(image)
    
images = np.array(images, dtype='uint8')

print(images[0].shape)

text_file = open("./external-traffic-signs/extResults.txt","r")
ylabel = np.zeros(size_images)
ylabel = text_file.readlines()
print(ylabel[0])

# Grayscaling
# imagesGray = np.zeros(shape = [images.shape[0],images.shape[1],images.shape[2]], dtype = np.int32)
# for i in range(size_images):
#    imagesGray[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
## I do not know why but cv2.cvtColor does not work for this example so I write it as
imagesGray = np.sum(images/3, axis=3, keepdims=True)
print(imagesGray.shape)

# Normalize
imgNormal = (imagesGray - 128)/128