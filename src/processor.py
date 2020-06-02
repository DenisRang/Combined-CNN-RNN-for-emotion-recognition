"""
Process an image that we can pass to our networks.
"""
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

def process_image(image, target_shape=(96, 96)):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w = target_shape
    image = load_img(image, color_mode='grayscale', target_size=(h, w))
    # image = load_img(image, color_mode='rgb',target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = img_arr

    return x
