from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    # Configurar CORS para permitir solicitudes desde cualquier origen
    CORS(app, resources={r"/images/*": {"origins": "*"}})
    CORS(app)

    from .routes import main
    app.register_blueprint(main)

    return app
