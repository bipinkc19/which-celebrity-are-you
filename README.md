# Which celebrityare you

Using Face Classification
Face classification is a simple package running on top of [face_recognition](https://pypi.org/project/face_recognition/) package which automates the training process and helps in easily getting the prediction. This could be a helpful addition to developers looking forward to adding the face recognition functionality in their projects.

It uses KNN algorithm and the model provided by face_recognition which spits out a vector of 128 length. Face classification has setup all the things to train and predict faces with optimum precision.

# Setup

```bash
pip install -r requirements.txt
```

# Example

```py
python3 find_look_alike_celeb.py
```
__Press 'c' to click picture. Only worlks when single person is infront of the cam__

# License

Apache License 2
