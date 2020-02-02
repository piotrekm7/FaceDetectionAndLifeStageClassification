import cv2
from keras.models import load_model


def resize_images(images):
    output = [cv2.resize(image, (50, 50)) for image in images]
    return output


def convert_data_to_float_and_normalize(data):
    data = data.astype('float32')
    data /= 255
    return data


def load_dnn_models():
    face_detection = load_face_detection_network()
    life_stage = load_life_stage_model()
    return face_detection, life_stage


def load_face_detection_network():
    net = cv2.dnn.readNetFromCaffe('models/caffe/deploy.prototxt',
                                   'models/caffe/res10_300x300_ssd_iter_140000.caffemodel')
    return net


def load_life_stage_model():
    model = load_model('models/life_stage_model.h5')
    return model
