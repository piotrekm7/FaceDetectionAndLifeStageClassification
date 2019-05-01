import cv2
import numpy as np
from utils import resize_images, convert_data_to_float_and_normalize


def face_detection(image, models):
    image, blob = prepare_image(image)
    net = models[0]

    detections = perform_detection(blob, net)
    filtered_detections = filter_detections(detections)

    for detection in filtered_detections:
        (height, width) = image.shape[:2]
        box = get_scaled_box(detection, width, height)
        draw_box_on_image(image, box, life_stage_model=models[1])

    return image


def prepare_image(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (0, 0, 0))
    return image, blob


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


def draw_box_on_image(image, box, life_stage_model):
    face = image[box[1]:box[3], box[0]:box[2]]
    class_id = get_life_stage_prediction(face, life_stage_model)
    color = get_color_for_class(class_id)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=3)


def get_life_stage_prediction(image, model):
    image = resize_images([image])[0]
    image = convert_data_to_float_and_normalize(image)
    image = np.expand_dims(image, axis=0)
    class_id = np.argmax(model.predict(image)[0])
    return class_id


def get_color_for_class(class_id):
    colors = {0: (255, 0, 0), 1: (0, 255, 0)}
    return colors[class_id]
