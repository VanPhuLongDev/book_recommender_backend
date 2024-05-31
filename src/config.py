import os 
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.environ.get("KEY")
basedir = os.path.abspath(os.path.dirname(__file__))
