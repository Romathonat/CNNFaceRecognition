import os

#to not display shell outpu from caffe
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
from PIL import Image


caffe.set_mode_cpu()

def is_face(image):
    #we use our CNN. the first argument is the architecture of our CNN, the second one are the weights
    net = caffe.Net('/datas/deploy.prototxt', '/datas/facenet_iter_200000.caffemodel', caffe.TEST)

    #we create a transformer that resize the shape of our datas
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # (36,36,1) -> (1,36,36), seen as (1,1,36,36)
    
    #we load our image, and we preprocess it
    #im = caffe.io.load_image(image_path, color=False)
    image_transformed = transformer.preprocess('data', image)

    #we give it as the data to our CNN
    net.blobs['data'].data[...] = image_transformed

    #we make the CNN work
    out = net.forward()

    #we get the class
    return out['prob'][0][1]
