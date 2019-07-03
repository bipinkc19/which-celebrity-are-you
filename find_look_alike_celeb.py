import os
import cv2
import pickle

from sklearn.metrics.pairwise import cosine_similarity

from face_recognition import face_locations, face_encodings

DIR = './celeberity_images'

def crop_face(img, boundings):
    y1, x2, y2, x1 = boundings[0]
    cropped_img = img[y1:y2, x1:x2]

    return cropped_img

def find_closest_image(embedding, celeb_to_embedding):
    highest_similarity = -100
    for celeb, celeb_embedding in zip(celeb_to_embedding.keys(), celeb_to_embedding.values()):
        similarity = cosine_similarity(embedding, celeb_embedding)
        if similarity > highest_similarity:
            highest_similarity = similarity
            similar_celeb = celeb
    print(highest_similarity, similar_celeb)
    return similar_celeb

celeb_to_embedding = pickle.load(open("./pickle/celebs_embeddings.pkl", "rb"))
camera = cv2.VideoCapture(0)

while True:
    return_value, img = camera.read()
    cv2.imshow('You', img)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        boundings = face_locations(img)
        if len(boundings) == 1:
            cropped_face = crop_face(img, boundings)
            embedding = face_encodings(cropped_face)
            if len(embedding) != 0:
                closest_celeb = find_closest_image(embedding, celeb_to_embedding)
                celeb_img = cv2.imread('./celeberity_images/' + closest_celeb + '.jpg')
                celeb_img = cv2.resize(celeb_img, (500, 500))
                cv2.imshow('Closest celeb', celeb_img)
