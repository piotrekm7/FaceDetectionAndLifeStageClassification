"""
Face detection with ResNet-10 model.

Typical usage:
    face_detection = Res10FaceDetection(model_filepath, proto_filepath)
    detections = face_detection(image)
"""

import cv2


class Res10FaceDetection:
    """
    Class for performing face detection with use of ResNet-10 model and opencv dnn library.

    ResNet-10 face detection model implemented in caffe was used.
    Model files can be downloaded from opencv github repository.

    Attributes:
        model_filepath: path to .caffemodel file
        proto_filepath: path to .prototxt file
    """

    def __init__(self, model_filepath, proto_filepath):
        """
        Inits Res10FaceDetection class.
        """
        self.net = cv2.dnn.readNetFromCaffe(proto_filepath, model_filepath)

    def __call__(self, image):
        """
        Runs network on given blob and returns detections
        Args:
            image: numpy 2d array with 3 channels, opencv image representation

        Returns:

        """
        blob = self.__prepare_image(image)
        self.net.setInput(blob)
        detections = self.net.forward()
        # Extract only important informations from output
        detections = detections[0, 0, :, 2:]
        return detections

    @staticmethod
    def __prepare_image(image):
        """
        Prepares image to detection with ResNet-10 as described in paper
        Args:
            image: numpy 2d array with 3 channels, opencv image representation

        Returns:
            opencv blob ready to use with ResNet-10 network
        """
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        return blob
