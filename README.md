# Face Classification

Face classification is a simple package running on top of [face_recognition](https://pypi.org/project/face_recognition/) package which automates the training process and helps in easily getting the prediction. This could be a helpful addition to developers looking forward to adding the face recognition functionality in their projects.

It uses KNN algorithm and the model provided by face_recognition which spits out a vector of 128 length. Face classification has setup all the things to train and predict faces with optimum precision.

## Installing

```bash
pip install face-classification
```

## Setting up locally

```bash
git clone git@gitlab.lftechnology.com:leapfrogai/face-classification.git
```

## Setting up requirements

```bash
pip install -r requirements.txt
```

## A simple example

```py
# To train create a directory with images stored in sub directories and the label as the folder name
from face_classification import train_model, FaceClassifier

train_model(train_dir='path_to/directories_containing/sub_folders_with_labels_as_folder_name', 
            model_file_name='model_dir'
)

# Initialize object with path of the saved model in above step
# It can also be initialized without any model to get boundings and embeddings only by not passing any arguements below.
face_classifier = FaceClassifier('model_dir')

# Get the predictions of all the faces in the image directly
predictions = face_classifier.predict(image)

# For more flexibility we can use the following functionality 
# Get the face boundings of faces in image
boundings = face_classifier.get_face_boundings(image)

# Get the embeddings of the face from the cropped face image from boundings
embeddings = face_classifier.get_face_embeddings(cropped_single_face_image)

# Get the prediciton from embeddings
predictions = face_classifier.classify_face(embeddings)
```

# License

Apache License 2
