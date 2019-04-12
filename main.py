from face import load_crop_resize
from face_embedding import get_embeddings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

khalesi = load_crop_resize('khalesi.jpg')
jhon = load_crop_resize('jhon.jpg')

me = load_crop_resize('aa.jpg')

images = [img for img in [me, khalesi, jhon]]
images = np.array(images)

for i in images:
    plt.imshow(i)
    plt.show()


ans = get_embeddings(images)

similarity = {}
for i, name in enumerate(['me', 'khalesi', 'jhon']):
    if name == 'me':
        continue
    sim = cosine_similarity([ans[0]], [ans[i]])
    similarity[name] = sim[0][0]

plt.bar(similarity.keys(), similarity.values(), color='g')
plt.show()

    




