from flask import Flask
from .controller.recommend_controller import recommend
from .extension import ma

def create_app(config_file = "config.py"):
    app = Flask(__name__)
    app.config.from_pyfile(config_file)
    ma.init_app(app)
    app.register_blueprint(recommend)
    return app
