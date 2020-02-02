"""
Performing face detection on image.

Typical usage:
    face_detection = FaceDetection(face_detection_model)
    bounding_boxes = face_detection(image)
"""

import numpy as np


class FaceDetection:
    """
    Class for performing face detection on an image.

    Attributes:
        face_detection_model: model for performing face detection, should return 2d numpy array
                with rows in format [Confidence, X_start, Y_start, X_end, Y_end]
    """

    def __init__(self, face_detection_model):
        """
        Inits FaceDetection class.
        """
        self.face_detection_model = face_detection_model

    def __call__(self, image):
        """
        Performs face detection on image.
        Args:
            image: numpy 2d array with 3 channels, opencv image representation

        Returns:
            Bounding boxes with faces, scaled to image.
        """
        detections = self.face_detection_model(image.copy())
        filtered_detections = self.__filter_detections(detections)

        height, width = image.shape[:2]
        boxes = [self.__get_scaled_box(detection, width, height) for detection in filtered_detections]
        return boxes

    @staticmethod
    def __filter_detections(detections, threshold=0.5):
        """
        Returns detections with confidence above given threshold.
        Args:
            detections: detections from face detection model
            threshold: minimum confidence to classify detection as correct

        Returns:
            List of detections above given confidence threshold.
        """
        filtered_detection = [detections[i] for i in range(0, len(detections)) if detections[i, 0] > threshold]
        return filtered_detection

    @staticmethod
    def __get_scaled_box(detection, image_width, image_height):
        """
        Scales bounding boxes to image dimensions
        Args:
            detection: detection from face detection model
            image_width: width of image
            image_height: height of image

        Returns:
        Bounding box containing face scaled to an image.
        """
        box = detection[1:5] * np.array([image_width, image_height, image_width, image_height])
        return box.astype('int')
