from src.util.model import predictTopK
from flask import jsonify
import os

def recommender(user_id, topK):
    listBooks = predictTopK(user_id, topK)
    listBooks = [int(book_id) for book_id in listBooks]
    return listBooks
    # return jsonify({
    #                 "listBooks": listBooks,
    #                 "message": "SUCCEESS",
    #                 "status": 1
    #             }), 201

