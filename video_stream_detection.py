"""
Detecting faces and predicts life stage on video.

Args:
    --camera: camera id or path to a video file

Typical use:
    python video_stream_detection.py --camera="video.avi"
"""

import argparse
import cv2
from single_image_detection import prepare_processing_engines


def parse_args():
    """
    Parse command line arguments.
    Returns:
        Parsed arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--camera', default=0, help='camera id or path to a video file')
    return vars(ap.parse_args())


def main():
    """
    Runs face detection and life stage classification on video and displays the results.
    """
    args = parse_args()
    cap = cv2.VideoCapture(args['camera'])
    image_processor = prepare_processing_engines()
    while True:
        _, frame = cap.read()
        processed_image = image_processor(frame)
        cv2.imshow('Output', processed_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
