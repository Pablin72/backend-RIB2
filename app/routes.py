from flask import Blueprint, request, jsonify, send_from_directory
import os
from .utils import extract_features, find_similar_images, show_images

main = Blueprint('main', __name__)

# Ruta para manejar la carga de archivos
@main.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        temp_path = os.path.join('/tmp', file.filename)
        file.save(temp_path)
        similar_images = find_similar_images(temp_path)
        show_images(temp_path, similar_images)
        os.remove(temp_path)

        return jsonify(similar_images), 200
    

@main.route('/images/<path:filename>')
def serve_image(filename):
    image_folder = '/Users/sebitas/EPN/Septimo-Semestre/Information-Retrieval/PROYECTO-RECONOCIMIENTO/BACKEND/backend-RIB2'
    
    # Asegurarse de que no haya duplicaciones en la ruta
    filename = filename.replace('101_ObjectCategories/101_ObjectCategories', '101_ObjectCategories')

    full_path = os.path.join(image_folder, filename)
    print(f"Serving image from: {full_path}")  # Agrega este log para verificar la ruta completa

    return send_from_directory(image_folder, filename)