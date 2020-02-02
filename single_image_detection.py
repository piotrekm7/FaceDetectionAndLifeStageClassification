"""
Detecting faces and predicts life stage on single image.

Args:
    --image: path to input image

Typical use:
    python single_image_detection.py --image="test.jpg"
"""

import argparse
import cv2
from face_detection_and_life_stage_classification import FaceDetectionAndLifeStageClassification
from face_detection import FaceDetection
from res10_face_detection import Res10FaceDetection
from life_stage_prediction import LifeStagePrediction


def parse_args():
    """
    Parse command line arguments.
    Returns:
        Parsed arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='path to input image')
    return vars(ap.parse_args())


def main():
    """
    Runs face detection and life stage classification on image and shows the results.
    """
    args = parse_args()
    image_path = args['image']
    image = cv2.imread(image_path)
    image_processor = prepare_processing_engines()
    processed_image = image_processor(image)
    cv2.imshow('Output', processed_image)
    cv2.waitKey(0)


def prepare_processing_engines():
    """
    Loads all machine learning models for processing an image.
    Returns:
        Image processor, which detects faces on image and classify their life stage.
    """
    res10_face_model = Res10FaceDetection('models/caffe/res10_300x300_ssd_iter_140000.caffemodel',
                                          'models/caffe/deploy.prototxt')
    face_detection_backend = FaceDetection(res10_face_model)
    life_stage_backend = LifeStagePrediction('models/life_stage_model.h5')
    image_processor = FaceDetectionAndLifeStageClassification(face_detection_backend, life_stage_backend)
    return image_processor


if __name__ == '__main__':
    main()
