from flask import Flask
from .controller.recommend_controller import recommend
from .config import dbUrl
from .service.connectDB import ConnectDB
from .util.phobert_model import Phobert
from .controller.books_controller import books


def create_app(config_file = "config.py"):
    app = Flask(__name__)
    ConnectDB.get_instance(dbUrl)
    Phobert.get_instance()
    app.register_blueprint(recommend)
    app.register_blueprint(books)
    return app
