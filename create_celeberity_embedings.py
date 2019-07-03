import os
import cv2
import pickle

from face_classification import FaceClassifier

DIR = './celeberity_images'
face_classifier = FaceClassifier(None)

def crop_face(img, boundings):
    y1, x2, y2, x1 = boundings[0]
    cropped_img = img[y1:y2, x1:x2]

    return cropped_img

celeb_to_embedding = {}
for image_dir in os.listdir(DIR):
    label = image_dir.split('.')[0]
    img = cv2.imread(os.path.join(DIR, image_dir))
    boundings = face_classifier.get_face_boundings(img)
    cropped_face = crop_face(img, boundings)
    embedding = face_classifier.get_face_embeddings(cropped_face)
    celeb_to_embedding[label] = embedding

pickle.dump(celeb_to_embedding, open("./pickle/celebs_embeddings.pkl", "wb"))
