
import caffe
import numpy as np
from PIL import Image

#we choose the mode (cpu or gpu)
caffe.set_mode_cpu()

#we use our CNN. the first argument is the architecture of our CNN, the second one are the weights
net = caffe.Net('/datas/deploy.prototxt', '/datas/facenet_iter_200000.caffemodel', caffe.TEST)

#we create a transformer that resize the shape of our datas
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # (36,36,1) -> (1,36,36), seen as (1,1,36,36)
 
#we load our image, and we preprocess it                          
im = caffe.io.load_image('/datas/train_images/1/Image000605.pgm', color=False)
image_transformed = transformer.preprocess('data', im)

#we give it as the data to our CNN
net.blobs['data'].data[...] = image_transformed

#we make the CNN work
out = net.forward()

#we get the class
print out['prob']

# result [[ 0.01239852  0.98760146]] -> the more likely class is the second one(1, face)

#we have to change the facenet_train_test.prototxt to deploy.prototxt. Then, we remove the train and test label, and put this one 
#layer {
#  name: "data"
#  type: "Input"
 # top: "data"
 # top: "label"
 # input_param { shape: {dim: 1 dim: 1 dim: 36 dim: 36 } }
#} 
# and at the end we specify that we want a classification as the output
# layer {
#  name: "prob"
#  type: "Softmax"
#  bottom: "fc8"
#  top: "prob"
#}

