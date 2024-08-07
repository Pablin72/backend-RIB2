from flask import Blueprint, request, jsonify, send_from_directory
from .utils import extract_features, find_similar_images, show_images
import os

main = Blueprint('main', __name__)

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
    image_folder = '/Users/sebitas/EPN/Septimo-Semestre/Information-Retrieval/PROYECTO-RECONOCIMIENTO/BACKEND/backend-RIB2/101_ObjectCategories'
    return send_from_directory(image_folder, filename)
