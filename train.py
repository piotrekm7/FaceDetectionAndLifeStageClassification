import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from utils import resize_images, convert_data_to_float_and_normalize


def train():
    x_train, x_test, y_train, y_test = prepare_data()
    model = create_model()

    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))
    model.save('models/life_stage_model.h5')


def prepare_data():
    adults = load_images('data/faces/adults')
    children = load_images('data/faces/children')

    x = adults + children
    x = np.array(resize_images(x))
    x = convert_data_to_float_and_normalize(x)
    y = [0 for _ in adults] + [1 for _ in children]
    # Convert output to one-hot matrices
    y = keras.utils.to_categorical(y, 2)

    # Returns (x_train, x_test, y_train, y_test)
    data = train_test_split(x, y, test_size=0.2)
    return data


def load_images(path: str):
    files = os.listdir(path)
    images = [cv2.imread(os.path.join(path, file)) for file in files]
    return images


def create_model():
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




