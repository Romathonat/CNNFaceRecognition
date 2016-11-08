from is_face import is_face
import numpy as np
import caffe
import skimage.io
import skimage.draw
import skimage.transform
import timeit

#TODO: command-line interface
#TODO: resize too big picture at the begining (too long for high quality)


#the step for the window
step = 5
width_image_network = 36
heigh_image_network = 36
resize_factor = 0.85
detection_min = 0.994

image = caffe.io.load_image('/datas/merica.jpg', color=False)

# #we normalize the picture
# for i in range(len(image)):
#     for j in range(len(image[i])):
#         image[i][j] = image[i][j]/255


def slide_window(image, number_resize):
    faces_positions = []

    #take care if the image is smaller than 36*36
    for j in range((image.shape[1] - heigh_image_network) / step):
        for i in range((image.shape[0] - width_image_network) / step):
            if(is_face(image[i*step:i*step+width_image_network,j*step:j*step+heigh_image_network,:]) > detection_min):
                faces_positions.append((i*step/resize_factor**number_resize,j*step/(resize_factor**number_resize), number_resize))



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

for i,j,number_resize in faces_positions:
    resize_square_factor = 1/(resize_factor**number_resize)

    row_coordinates = np.array([i, i+width_image_network*resize_square_factor, i+width_image_network*resize_square_factor, i])
    col_coordinates = np.array([j, j, j+heigh_image_network*resize_square_factor, j+heigh_image_network*resize_square_factor])


    #we get the coordinates of the square, and we display a white square on the image
    rr, cc = skimage.draw.polygon_perimeter(row_coordinates, col_coordinates)
    image[rr,cc,:] = 1

#we save the image, and delete the last dimension (W*H for images with no colors)
skimage.io.imsave('/datas/output.jpg', np.squeeze(image, axis=(2,)))

print faces_positions
