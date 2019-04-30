import cv2
import numpy as np


def face_detection(image):
    image, blob = prepare_image(image)
    net = load_network()

    detections = perform_detection(blob, net)
    filtered_detections = filter_detections(detections)

    for detection in filtered_detections:
        (height, width) = image.shape[:2]
        box = get_scaled_box(detection, width, height)
        draw_box_on_image(image, box)

    return image


def prepare_image(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    return image, blob


def load_network():
    net = cv2.dnn.readNetFromCaffe('models/caffe/deploy.prototxt.txt',
                                   'models/caffe/res10_300x300_ssd_iter_140000.caffemodel')
    return net


def perform_detection(blob, net):
    net.setInput(blob)
    detections = net.forward()
    return detections[0, 0]


def filter_detections(detections, threshold=0.5):
    # Return detections with confidence above given threshold
    filtered_detections = [detections[i] for i in range(0, len(detections)) if detections[i, 2] > threshold]
    return filtered_detections


def get_scaled_box(detection, image_width, image_height):
    box = detection[3:7] * np.array([image_width, image_height, image_width, image_height])
    return box.astype('int')


def draw_box_on_image(image, box):
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=3)
