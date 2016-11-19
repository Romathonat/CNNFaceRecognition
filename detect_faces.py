from is_face import is_face
import numpy as np
import caffe
import skimage.io
import skimage.draw
import skimage.transform
import timeit

from DBSCAN.draft_dbscan import clustering_image

#TODO: command-line interface
#TODO: resize too big picture at the begining (too long for high quality)


#the step for the window
step = 5
width_image_network = 36
heigh_image_network = 36
resize_factor = 0.85
detection_min = 0.997
image_path = '/datas/output_generated/merica.jpg'
output_path = '/datas/output_generated'
caffemodel = 'facenet_iter_1400000.caffemodel'
image = caffe.io.load_image(image_path, color=False)

def slide_window(image, number_resize):
    faces_positions = []

    #take care if the image is smaller than 36*36
    for j in range((image.shape[1] - heigh_image_network) / step):
        for i in range((image.shape[0] - width_image_network) / step):
            score = is_face(image[i*step:i*step+width_image_network,\
                j*step:j*step+heigh_image_network,:], caffemodel)
            if(score > detection_min):
                faces_positions.append((i*step/resize_factor**number_resize,j*step/(resize_factor**number_resize), number_resize, score))

    return faces_positions

#we make a deep copy of the image to not alterate the source
image_copy = image[...]
faces_positions = []
count_resize = 0

while(image_copy.shape[0] >= 36 and image_copy.shape[1] >= 36):
    start_time = timeit.default_timer()
    faces_positions += slide_window(image_copy, count_resize)

    output_shape = (int(image_copy.shape[0]*resize_factor), int(image_copy.shape[1]*resize_factor))
    image_copy = skimage.transform.resize(image_copy,output_shape)

    print('Resize factor :{}, took {}'.format(count_resize, timeit.default_timer() - start_time))

    count_resize += 1

#we draw shapes where we detected faces

for i,j,number_resize, score in faces_positions:
    resize_square_factor = 1/(resize_factor**number_resize)

    row_coordinates = np.array([i, i+width_image_network*resize_square_factor, \
        i+width_image_network*resize_square_factor, i])
    col_coordinates = np.array([j, j, j+heigh_image_network*resize_square_factor,\
        j+heigh_image_network*resize_square_factor])


    #we get the coordinates of the square, and we display a white square on the image
    rr, cc = skimage.draw.polygon_perimeter(row_coordinates, col_coordinates)
    image[rr,cc,:] = 1

#we save the image, and delete the last dimension (W*H for images with no colors)
skimage.io.imsave('/datas/output_generated/output.jpg', np.squeeze(image, axis=(2,)))

#we use dbscan to get faces
cnn_detections = []
for x,y,number_resize, score in faces_positions:
    resize_square_factor = 1/(resize_factor**number_resize)
    cnn_detections.append('{} {} {} {} {}'.format(x,y, width_image_network*\
        resize_square_factor, heigh_image_network*resize_square_factor, score))

clustering_image(cnn_detections, image_path)
