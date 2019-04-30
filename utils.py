import cv2


def resize_images(images):
    output = [cv2.resize(image, (50, 50)) for image in images]
    return output


def convert_data_to_float_and_normalize(data):
    data = data.astype('float32')
    data /= 255
    return data
