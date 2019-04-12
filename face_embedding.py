from keras.models import load_model
from keras import Model
import numpy as np
 
model = load_model('./model/facenet_keras.h5')

# returning AvgPool (GlobalAveragePooling2D (None, 1792))     
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[-2].output)

def get_embeddings(img):

    embeddings = intermediate_layer_model.predict(img)
    # 3 is the number of images to predict
    return(np.reshape(embeddings, (-1, 3)))
