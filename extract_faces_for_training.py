import cv2
import face_detection
import os


def main():
    parent_folder = 'data/people'
    destination_folder = 'data/faces'
    folders = ('adults', 'children')
    for folder in folders:
        path = os.path.join(parent_folder, folder)
        files = os.listdir(path)
        for file in files:
            image = cv2.imread(os.path.join(path, file))
            output = get_face_with_highest_score(image)
            destination_path = os.path.join(destination_folder, folder)
            cv2.imwrite(os.path.join(destination_path, file), output)


def get_face_with_highest_score(image):
    image, blob = face_detection.prepare_image(image)
    net = face_detection.load_network()

    detections = face_detection.perform_detection(blob, net)

    (height, width) = image.shape[:2]
    box = face_detection.get_scaled_box(detections[0], width, height)
    output = image[box[1]:box[3], box[0]:box[2]]

    return output


if __name__ == '__main__':
    main()