"""
Predicts if face belongs to adult or child

Typical usage:
life_stage_predictor = LifeStagePrediction(model_path)
class = life_stage_predictor(image)
"""

import cv2
import numpy as np
import keras.models


class LifeStagePrediction:
    """
    Class for predicting if face on an image belongs to adult or child.

    Attributes:
        life_stage_model_path: filepath for keras model weights
    """

    def __init__(self, life_stage_model_path):
        """
        Inits LifeStagePrediction class.
        """
        self.life_stage_model = keras.models.load_model(life_stage_model_path)

    def __call__(self, image):
        """
        Runs model on face image and predicts if the face belongs to adult or child
        Args:
            image: numpy 2d array with 3 channels, opencv image representation

        Returns:
            0 if adult face, 1 if child face
        """
        image = self.__prepare_image(image.copy())
        class_id = np.argmax(self.life_stage_model.predict(image)[0])
        return class_id

    def __prepare_image(self, image):
        """
        Prepares image for neural network.
        Args:
            image: numpy 2d array with 3 channels, opencv image representation

        Returns:
            Processed image, ready to use with neural network.
        """
        image = self.__resize_image(image)
        image = self.__convert_data_to_float_and_normalize(image)
        image = np.expand_dims(image, axis=0)
        return image

    @staticmethod
    def __resize_image(image):
        """
        Resizes image to 50x50 since these dimension are required by neural network.
        Args:
            image: image to resize

        Returns:
            Resized image.
        """
        output = cv2.resize(image, (50, 50))
        return output

    @staticmethod
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
