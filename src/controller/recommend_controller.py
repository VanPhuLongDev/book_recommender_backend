from flask import Blueprint, request
from src.service.recommend_service import recommender
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os

recommend = Blueprint('recommend', __name__, url_prefix='/recommend')

base_path = os.path.dirname(__file__)
full_path = os.path.join(base_path, 'data_150k.txt')
rating_df = pd.read_csv(full_path, delimiter='\t', header=None, names=['userId', 'bookId', 'rating'])
filtered_rating_df = rating_df[rating_df['rating'] >= 3]
data = filtered_rating_df[["userId","bookId"]]
user_book_matrix = data.pivot(index='userId', columns='bookId', values='bookId').notna().astype(int)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_book_matrix)

def find_similar_user(book_ids, model, user_book_matrix):
    book_vector = np.zeros(user_book_matrix.shape[1])
    for book_id in book_ids:
        if book_id in user_book_matrix.columns:
            book_vector[user_book_matrix.columns.get_loc(book_id)] = 1
    book_vector = book_vector.reshape(1, -1)

    # Find the nearest neighbors
    distances, indices = model.kneighbors(book_vector, n_neighbors=2)
    similar_user_id = user_book_matrix.index[indices.flatten()[0]]

    return similar_user_id
# books = [ 34330, 23107, 13419]
# similar_user_id = find_similar_user(books, model_knn, user_book_matrix)

@recommend.route('/predict-top-K', methods=['POST'])
def predictTopK():
    data = request.get_json()
    bookIds = data.get('book_ids')
    similar_user_id = find_similar_user(bookIds,model_knn,user_book_matrix)
    topK= data.get('topK')
    print(similar_user_id, topK)
    return recommender(similar_user_id, topK)


# @recommend.route('/predict-convmf', methods=['POST'])
# def predictTopK_convmf():
#     data = request.get_json()
#     bookIds = data.get('book_ids')
#     similar_user_id = find_similar_user(bookIds,model_knn,user_book_matrix)
#     topK= data.get('topK')
#     print(similar_user_id, topK)
#     return recommender_convmf(similar_user_id, topK)
