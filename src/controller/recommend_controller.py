from flask import Blueprint, request
from src.service.recommend_service import recommender

recommend = Blueprint('recommend', __name__, url_prefix='/recommend')

@recommend.route('/predict-top-K', methods=['POST'])
def predictTopK():
    data = request.get_json()
    userId = data.get('user_id')
    topK= data.get('topK')
    print(userId, topK)
    return recommender(userId, topK)
