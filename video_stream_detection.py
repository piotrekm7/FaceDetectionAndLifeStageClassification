import argparse
import cv2
from face_detection import face_detection
from utils import load_dnn_models


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--camera', default=0, help='camera id or path to a video file')
    return vars(ap.parse_args())


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args['camera'])
    dnn_models = load_dnn_models()
    while True:
        _, frame = cap.read()
        detection = face_detection(frame, dnn_models)
        cv2.imshow('Output', detection)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
