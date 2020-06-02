"""
Visualize different augmentations of preprocessed Aff-Wild2 frame
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2

image_path = '/Users/denisrangulov/Fake Colab/cropped_aligned/79-30-960x720/01194.jpg'
image = load_img(image_path, color_mode='grayscale', target_size=(96, 96))
image = img_to_array(image)

image_gen = ImageDataGenerator(rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=False,
                               fill_mode='reflect',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5])

cv2.imwrite('/Users/denisrangulov/Google Drive/EmotionRecognition/figures/origin.png', image)
datagen = ImageDataGenerator()

aug_brightness = datagen.apply_transform(x=image, transform_parameters={'brightness': 1.5})
cv2.imwrite('/Users/denisrangulov/Google Drive/EmotionRecognition/figures/aug_brightness.png', aug_brightness)

aug_rotation = datagen.apply_transform(x=image, transform_parameters={'theta': 15})
cv2.imwrite('/Users/denisrangulov/Google Drive/EmotionRecognition/figures/aug_rotation.png', aug_rotation)

aug_shift = datagen.apply_transform(x=image, transform_parameters={'tx': 5, 'ty': 5})
cv2.imwrite('/Users/denisrangulov/Google Drive/EmotionRecognition/figures/aug_shift.png', aug_shift)

aug_shear = datagen.apply_transform(x=image, transform_parameters={'shear': 5})
cv2.imwrite('/Users/denisrangulov/Google Drive/EmotionRecognition/figures/aug_shear.png', aug_shear)

aug_zoom = datagen.apply_transform(x=image, transform_parameters={'zx': 1.25, 'zy': 1.25})
cv2.imwrite('/Users/denisrangulov/Google Drive/EmotionRecognition/figures/aug_zoom.png', aug_zoom)

aug_flip_horizontal = datagen.apply_transform(x=image, transform_parameters={'flip_horizontal': True})
cv2.imwrite('/Users/denisrangulov/Google Drive/EmotionRecognition/figures/aug_flip_horizontal.png', aug_flip_horizontal)
