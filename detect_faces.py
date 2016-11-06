from is_face import is_face
import numpy as np
import caffe
import skimage.io
import skimage.draw
import matplotlib.pyplot as pyplot

#TODO: command-line interface

step = 10
width_image_network = 36
heigh_image_network = 36

image = caffe.io.load_image('/datas/merica.jpg', color=False)

#the step for the window
faces_positions = []

#TODO: take care if the image is smaller than 36*36
for j in range((image.shape[1] - heigh_image_network) / step):
    for i in range((image.shape[0] - width_image_network) / step):
        if(is_face(image[i*step:i*step+width_image_network,j*step:j*step+heigh_image_network,:]) > 0.97):
            faces_positions.append((i*step,j*step))

#we make a deep copy of the image to not alterate the source
image_copy = image[...]

#we draw shapes where we detected faces

for i,j in faces_positions:
    row_coordinates = np.array([i, i+width_image_network, i+width_image_network, i])
    col_coordinates = np.array([j, j, j+heigh_image_network, j+heigh_image_network])

    #we get the coordinates of the square, and we display a white square on the image
    rr, cc = skimage.draw.polygon_perimeter(row_coordinates, col_coordinates)
    image_copy[rr,cc,:] = 1

#we save the image, and delete the last dimension (W*H for images with no colors)
skimage.io.imsave('/datas/output.jpg', np.squeeze(image_copy, axis=(2,)))

print faces_positions
