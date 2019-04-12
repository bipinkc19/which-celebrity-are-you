import cv2
import matplotlib.pyplot as plt

detector = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

def get_cropped_face(img):

    rects = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=13, minSize=(40, 40))
    b_boxes = [(x, y, x + w, y + h) for (x, y, w, h) in rects]
    for b_box in b_boxes:
        y_img = img[b_box[1]:b_box[3], b_box[0]:b_box[2]]
        break

    return y_img


def load_crop_resize(filename):

    img = plt.imread(filename)
    cropped = get_cropped_face(img) 
    resized = cv2.resize(cropped, (160, 160))
    
    return resized
