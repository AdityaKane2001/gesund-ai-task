from pprint import pprint
import tensorflow as tf
from imutils import paths
import streamlit as st
import numpy as np

from format import visualize_results


def parse_label_mapping(args_to_names): 
    mappings = args_to_names.getvalue().decode("utf-8").split("\n")
    mappings = [pair.strip("\n") for pair in mappings]
    mappings = [pair.split(" : ") for pair in mappings]
    mappings = [(int(pair[0]), pair[1]) for pair in mappings]
    mappings = {key:value for (key, value) in mappings}
    return mappings

    
def get_results():
    pass

def reset_exp():
    pass


def parse_image(imagepath):
    # st.write()
    image_string = imagepath.getvalue()
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[224, 224])
    image = image * 255.
#     label = label_mapping[tf.strings.split(tf.strings.split(image_path, sep="/")[-1], sep="_")[0]]
    return image, imagepath.name

def parse_labels(labels_file, mapping):
    all_labels = labels_file.getvalue().decode("utf-8").split("\n")
    all_labels = [label.strip("\r").split(" : ") for label in all_labels]
    all_labels = {key:int(mapping[value]) for key, value in all_labels}
    return all_labels
    

def get_predict_images(image_list):
    return [parse_image(imagepath)[0] for imagepath in image_list] 


def get_eval_images(image_list):
    return [parse_image(imagepath) for imagepath in image_list]


# def get_labels(labels_file, label_mapping):
#     # all_labels = labels_file.getvalue().decode("utf-8").split("\n")
#     return [tf.one_hot(label_mapping[imagepath.split("/")[-1].split("_")[0]], depth=5) for imagepath in image_list]

def get_predict_ds(images, args_to_names):
    mappings = parse_label_mapping(args_to_names)
    print(mappings)
    all_images = np.array(get_predict_images(images))
    all_images = all_images[np.newaxis,:]
    all_images_ds = tf.data.Dataset.from_tensor_slices(all_images)
    return all_images_ds, mappings


def get_eval_ds(images, labels, args_to_names):
    int_to_label_mappings = parse_label_mapping(args_to_names)
    label_to_int_mapping = {v: k for k, v in int_to_label_mappings.items()}
    print(label_to_int_mapping)
    print(int_to_label_mappings)
    
    images_and_imagenames = get_eval_images(images)
    
    images, imagenames = list(zip(*images_and_imagenames))
    label_mappings = parse_labels(labels, label_to_int_mapping)
    labels = [label_mappings[imagename]
              for imagename in imagenames]

    all_images = np.array(images)
    # all_images = all_images[:]
    all_images_ds = tf.data.Dataset.from_tensor_slices(all_images)

    all_labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    all_labels_ds = all_labels_ds.map(lambda x: tf.one_hot(x, depth=5, dtype=tf.float32))

    ds = tf.data.Dataset.zip((all_images_ds, all_labels_ds))
    return ds, int_to_label_mappings
