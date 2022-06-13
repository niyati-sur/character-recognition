import mnist
import scipy.misc
from PIL import Image

def show_image(image_as_2d_array):
    im = Image.fromarray(image_as_2d_array)
    im.show()

def show_multiple_images(images_as_2d_array, how_many):
    from matplotlib import pyplot
    for i in range(how_many):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(images_as_2d_array[i], cmap=pyplot.get_cmap('gray'))
        pyplot.show()

train_images = mnist.train_images()
print(train_images.shape)
train_labels = mnist.train_labels()
print(train_labels.shape)
print(train_labels[0])

#flip background of images
import numpy as np
train_images = train_images*-1+256
train_images_flatten = train_images.reshape(train_images.shape[0], 28*28)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(train_images_flatten, train_labels)

test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(test_images.shape)
print(test_labels.shape)

import numpy as np

#get predicted values
test_images=test_images*-1+256
test_images_flatten = test_images.reshape(test_images.shape[0], 28*28)

y_test_pred = knn.predict(test_images_flatten)
print(y_test_pred[0:10])

#verify test images with actual data
print(test_labels[0:10])

#check accuracy of your model
from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(test_labels, y_test_pred)
print(accuracy)

# project 1
# use class data to read a image
# change RGB to grey, so that it's size is changed from 128*128*3 to 128*128*1
# reshape image to 28*28*1 size
# try predicing a value, check if it's working correctly
# where could it go wrong?
# What could be reasons?
# (hint) which of the following cases one would better serve? fitting with too much data, or less data?

# project 2
# scan a random image and break it up into several images, where each image contains a single letter.
# repeat the steps from project 1 to see if you can predict each letter correctly.