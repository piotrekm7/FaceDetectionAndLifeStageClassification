"""
Training model for life stage classification.
"""

import os
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import cv2
from sklearn.model_selection import train_test_split
import numpy as np


def train():
    """
    Runs training process.
    """
    x_train, x_test, y_train, y_test = __prepare_data()
    model = __create_model()

    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))
    model.save('models/life_stage_model.h5')


def __prepare_data():
    """
    Prepares data for neural network.
    Returns:
        Prepared data.
    """
    adults = __load_images('data/faces/adults')
    children = __load_images('data/faces/children')

    x = adults + children
    x = np.array(__resize_images(x))
    x = __convert_data_to_float_and_normalize(x)
    y = [0 for _ in adults] + [1 for _ in children]
    y = keras.utils.to_categorical(y, 2)

    data = train_test_split(x, y, test_size=0.2)
    return data


def __resize_images(images):
    """
    Resizes images to 50x50 dimension.
    Args:
        images: list of images in opencv format

    Returns:
        List of resized images.
    """
    output = [cv2.resize(image, (50, 50)) for image in images]
    return output


def __convert_data_to_float_and_normalize(data):
    """
    Converts int numpy array to float and normalizes it.
    Args:
        data: numpy array

    Returns:
        Converted to float and normalized numpy array
    """
    data = data.astype('float32')
    data /= 255
    return data


def __load_images(path):
    """
    Loads images from hard drive.
    Args:
        path: Path to folder containing images

    Returns:
        List of images from specified folder in opencv format.
    """
    files = os.listdir(path)
    images = [cv2.imread(os.path.join(path, file)) for file in files]
    return images


def __create_model():
    """
    Creates neural network model.
    Returns:
        Neural network model.
    """
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(50, 50, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model


if __name__ == '__main__':
    train()




