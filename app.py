from src import create_app
import numpy as np
import inspect

from train_convmf_vn import load_model
model = load_model()

if __name__ == '__main__':
    app = create_app()
    app.model = model
    app.run(debug = True)