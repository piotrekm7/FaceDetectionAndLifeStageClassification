import argparse
import cv2
from face_detection import face_detection
from utils import load_dnn_models


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='path to input image')
    return vars(ap.parse_args())


def main():
    args = parse_args()
    image_path = args['image']
    image = cv2.imread(image_path)
    dnn_models = load_dnn_models()
    detection = face_detection(image, dnn_models)
    cv2.imshow('Output', detection)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
