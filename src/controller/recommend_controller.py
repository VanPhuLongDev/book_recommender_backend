from flask import Blueprint, request,current_app,jsonify
from src.service.recommend_service import recommender
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os

recommend = Blueprint('recommend', __name__, url_prefix='/recommend')

base_path = os.path.dirname(__file__)
full_path = os.path.join(base_path, 'data.txt')
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
    results = None
    try:
        results = recommender(similar_user_id, topK)
    except Exception as e:
        print (e)
    return jsonify({
                    "listBooks": results,
                    "message": "SUCCEESS",
                    "status": 1
                }), 201


@recommend.route('/predict-convmf', methods=['POST'])
def predictTopK_convmf():
    model = current_app.model
    data = request.get_json()
    bookIds = data.get('book_ids')
    similar_user_id = find_similar_user(bookIds,model_knn,user_book_matrix)
    topK= data.get('topK')
    print('top k :', topK)
    print(similar_user_id, topK)
    try:
        recs,score_item = model.recommend(user_id=str(similar_user_id), k=topK)
        print('recs controller: ',recs)
        return jsonify({
                        "listBooks": recs,
                        "message": "SUCCEESS",
                        "status": 1
                    }), 201
    except Exception as e:
        print('error in convmf recommend: ',e)
        return None

def combine_recommend(a, b):
    # Tìm các phần tử chung
    common_elements = list(set(a) & set(b))
    
    # Loại bỏ các phần tử chung khỏi hai mảng ban đầu
    a_unique = [x for x in a if x not in common_elements]
    b_unique = [x for x in b if x not in common_elements]
    
    # Tạo danh sách kết hợp từ các phần tử chung
    result = common_elements[:]
    
    # Thêm các phần tử không chung xen kẽ nhau
    len_a = len(a_unique)
    len_b = len(b_unique)
    max_len = max(len_a, len_b)
    
    for i in range(max_len):
        if i < len_a:
            result.append(a_unique[i])
        if i < len_b:
            result.append(b_unique[i])
    
    return result

@recommend.route('/hybrid_recommend', methods=['POST'])
def hybrid_recommend():
    model = current_app.model
    data = request.get_json()
    bookIds = data.get('book_ids')
    similar_user_id = find_similar_user(bookIds,model_knn,user_book_matrix)
    topK= data.get('topK')
    try:
        lightcgn = recommender(similar_user_id, topK)
        recs,score_item = model.recommend(user_id=str(similar_user_id), k=topK)
        recs_int = [int(i) for i in recs]
        result = combine_recommend(recs_int,lightcgn)
        print('result: ',result)
        value =  jsonify({
                        "listBooks": result,
                        "lightcgn": lightcgn,
                        "convmf": recs_int,
                        "score_convmf": score_item.tolist(),
                        "message": "SUCCEESS",
                        "status": 1
                    }), 201
        return value
    except Exception as e:
        print('error in hybrid recommend system: ', e)
        return None
