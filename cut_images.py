from __future__ import division
import math
import os
import skimage.io
import skimage.draw
import skimage.transform
from PIL import Image

def slice_from_path(image_path, out_name, outdir, slice_size):
    img = Image.open(image_path)
    slice(img, out_name, outdir, slice_size)


def slice(img, out_name, outdir, slice_size):
    """slice an image into parts slice_size tall"""
    width, height = img.size
    upper = 0
    left = 0
    horizontal_slices = int(math.ceil(height/slice_size))
    vertical_slices = int(math.ceil(width/slice_size))

    horizontal_count = 1
    vertical_count = 1
    for horizontal_slice in range(horizontal_slices):
        #if we are at the end, set the lower bound to be the bottom of the image
        if horizontal_count == horizontal_slices:
            lower_height = height
            break
        else:
            lower_height = int(horizontal_count * slice_size)

        for vertical_slice in range(vertical_slices):
            #if we are at the end, set the lower bound to be the bottom of the image
            if vertical_count == vertical_slice:
                lower_width = width
                break
            else:
                lower_width = int(vertical_count * slice_size)
            #set the bounding box! The important bit     
            bbox = (left, upper, lower_width, lower_height)
            working_slice = img.crop(bbox)
            left += slice_size
            #save the slice
            working_slice.save(os.path.join(outdir, "slice_" + out_name + "_" + str(horizontal_count)+"_"+str(vertical_count)+".png"))
            vertical_count +=1

        upper += slice_size
        left = 0
        vertical_count = 1
        horizontal_count += 1

def pyramidal_slice(image_path, out_name, outdir, slice_size, resize_factor = 0.85):
    size = 36
    count_resize = 0
    image_copy = Image.open(image_path)
    slice(image_copy, out_name+str(count_resize), outdir, slice_size)
    while(image_copy.width >= size and image_copy.height >= size):

        output_shape = (int(image_copy.width*resize_factor), int(image_copy.height*resize_factor))
        image_copy = image_copy.resize(output_shape, Image.ANTIALIAS)
        slice(image_copy,out_name+str(count_resize), outdir, slice_size)
        count_resize += 1

def pyramidal_slice_folder(outdir, slice_size):
    directory = os.path.join(outdir,"./output_thumbnail")
    if not os.path.exists(directory):
        os.makedirs(directory)

    for file in os.listdir(outdir):
        if file.endswith(".jpg"):
            print(file + " is being sliced...")
            pyramidal_slice(file, file, os.path.join(outdir,"./output_thumbnail"), slice_size)
            print(file + " got sliced!")



def test_slice():
    slice_from_path("merica.jpg","merica", os.getcwd(), 32)

def test_pyramidal_slice():
    pyramidal_slice("merica.jpg","merica", os.getcwd(), 32)


pyramidal_slice_folder(".", 32)