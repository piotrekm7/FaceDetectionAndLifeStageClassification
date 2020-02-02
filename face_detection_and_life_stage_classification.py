"""
Performing face detection and life stage classification on image

Typical use:
    engine = FaceDetectionAndLifeStageClassification()
    processed_image = engine(image)
"""

import cv2


class FaceDetectionAndLifeStageClassification:
    """
    A class for performing face detection and life stage classification on image.

    Attributes:
        face_detection_model: model for detecting faces on image
        life_stage_model: model for predicting life stage of face
    """

    def __init__(self, face_detection_model, life_stage_model):
        """
        Inits FaceDetectionAndLifeStageClassification class.
        """
        self.face_detection_model = face_detection_model
        self.life_stage_model = life_stage_model

    def __call__(self, image):
        """
        Performs face detection and life stage classification on image.
        Args:
            image: numpy 2d array with 3 channels, opencv image representation

        Returns:
            Processed image with colorful rectangles around faces.
            Rectangle color depends on face life stage.
        """
        image = image.copy()
        boxes = self.face_detection_model(image)
        for box in boxes:
            class_id = self.__predict_life_stage(image, box)
            color = self.__get_color_for_class(class_id)
            self.__draw_box_on_image(image, box, color)

        return image

    def __predict_life_stage(self, image, box):
        """
        Extracts face from image and passes to life stage classifier.
        Args:
            image: numpy 2d array with 3 channels, opencv image representation
            box: face bounding box

        Returns:
            Life stage class prediction for given face.
        """
        face = image[box[1]:box[3], box[0]:box[2]]
        return self.life_stage_model(face)

    @staticmethod
    def __get_color_for_class(class_id):
        """
        Specifies color of box to be draw on the image for particular life stage class.
        Args:
            class_id: life stage class id

        Returns:
            Blue for adults and green for kids. Color format is BGR.
        """
        colors = {0: (255, 0, 0), 1: (0, 255, 0)}
        return colors[class_id]

    @staticmethod
    def __draw_box_on_image(image, box, color):
        """
        Draws rectangle border containing face on image.
        Args:
            image: numpy 2d array with 3 channels, opencv image representation
            box: face bounding box
            color: desired rectangle border color

        Returns:
            Processed image with rectangles on it.
        """
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=3)
