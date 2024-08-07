import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend
import matplotlib.pyplot as plt

# Cargar el modelo y el archivo de características
local_weights_file = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(weights=local_weights_file, include_top=False, input_shape=(224, 224, 3))
model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

with open('index.pkl', 'rb') as f:
    index = pickle.load(f)

image_paths = list(index.keys())
features = np.array([index[path][0] for path in image_paths])
labels = [index[path][1] for path in image_paths]

nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(features)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

def extract_features(model, image_path):
    img_array = preprocess_image(image_path)
    features = model.predict(img_array)
    return features.flatten()

def find_similar_images(query_image_path):
    query_features = extract_features(model, query_image_path)
    distances, indices = nbrs.kneighbors([query_features])
    
    similar_images = []
    for i, idx in enumerate(indices[0]):
        path = image_paths[idx]
        # Asegurarse de que no haya duplicaciones en la ruta
        path = path.replace('101_ObjectCategories/101_ObjectCategories', '101_ObjectCategories')
        similar_images.append((path, distances[0][i]))

    return similar_images



def show_images(query_image_path, similar_images):
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    query_img = load_img(query_image_path, target_size=(224, 224))
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    for i, (img_path, dist) in enumerate(similar_images):
        similar_img = load_img(img_path, target_size=(224, 224))
        axes[i + 1].imshow(similar_img)
        axes[i + 1].set_title(f'Similar {i+1}\nDist: {dist:.2f}')
        axes[i + 1].axis('off')

    plt.show()
