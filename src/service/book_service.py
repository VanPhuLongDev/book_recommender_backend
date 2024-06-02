from src.util.phobert_model import Phobert
from .connectDB import ConnectDB
class BookService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.collection = ConnectDB.get_instance().db["books"]


    def create_book(self, book_data):
        id = book_data["id"]
        description = book_data["description"]
        embedding = Phobert.get_instance().get_embedding(description)
        
        result = self.collection.insert_one({"_id": id, "embedding": embedding})

        return result.inserted_id

    def get_book(self, book_id):
        book = self.collection.find_one({"_id": book_id})
        return book

    def update_book(self, book_id, updated_data):
        result = self.collection.update_one({"_id": book_id}, {"$set": updated_data})
        return result.modified_count

    def delete_book(self, book_id):
        result = self.collection.delete_one({"_id": book_id})
        return result.deleted_count

    def get_relative_books(self, book_id,k=10):
        book = self.collection.find_one({"_id": book_id})
        embedding = book["embedding"]

        query = [
            {
                "$vectorSearch": {
                        "index": "vector_index",
                        "queryVector": embedding,
                        "path": "embedding",
                        "numCandidates": 150,
                        "limit": k,
                    },
            },
            {
                "$unset":"embedding",
            },
            {
                "$project": {
                    "_id": 1,
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }
        ]
        books = self.collection.aggregate(query)
        return list(books)
    
    def get_semantic_search(self, texts,k=10):
        embedding = Phobert.get_instance().get_embedding(texts)

        query = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": embedding,
                    "path": "embedding",
                    "numCandidates": 150,
                    "limit": k}
            },
            {
                "$unset":"embedding",
            },
            {
                "$project": {
                    "_id": 1,
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }
        ]

        books = self.collection.aggregate(query)
        return list(books)

    def close_connection(self):
        self.client.close()