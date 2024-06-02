# from pymongo import MongoClient
from pymongo.mongo_client import MongoClient

class ConnectDB:

    instance = None

    @staticmethod
    def get_instance(url=""):
        if ConnectDB.instance is None:
            ConnectDB.instance = ConnectDB(url)
        return ConnectDB.instance

    def __init__(self, url):
        try:
            self.client = MongoClient(url)
            self.db = self.client["main_db"]
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)


    def close(self):
        self.client.close()

