import schedule
import time
import os
import logging
from trainModelLightGCN import train_model_LGCN
import requests
from train_convmf_vn import train_model

log_file_path = 'daily_task.log'
logging.basicConfig(
    filename=log_file_path, 
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    encoding='utf-8'
)


def call_api(page):
    url = f'https://bookrecommendation.website/api/rating-recommendation?page={page}&size=1000'
    headers = {
        'User-Agent': 'requests'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def save_data():
    data = []
    try:
        for i in range(100):
            print('i = ', i)
            result = call_api(i)
            data.extend(result['content'])
        file_name = 'data.txt'
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, 'w', encoding='utf-8') as file:
            for item in data:
                line = f"{item['userId']}\t{item['bookId']}\t{item['ratingRecommendation']}\n"
                file.write(line)

        logging.info('Dữ liệu đã được ghi vào file thành công!')
    
    except Exception as e:
        logging.info(f'Có lỗi trong quá trình lưu data: {e}')

def train_model_Light_GCN():
    logging.info("Bắt đầu huấn luyện mô hình LightGCN.")
    print("Bắt đầu huấn luyện mô hình LightGCN.")
    
    try:
        train_model_LGCN()
        logging.info("Huấn luyện mô hình LightGCN thành công.")
        print("Huấn luyện mô hình LightGCN thành công.")
    except Exception as e:
        logging.error(f"Lỗi trong quá trình huấn luyện mô hình LightGCN: {e}")
        print(f"Lỗi trong quá trình huấn luyện mô hình LightGCN: {e}")

def train_model_convmf():
    logging.info("Bắt đầu huấn luyện mô hình 2.")
    print("Bắt đầu huấn luyện mô hình 2.")
    
    try:
        # Giả lập quá trình huấn luyện mô hình với thời gian chạy 5 giây
        # time.sleep(5)
        train_model()
        logging.info("Huấn luyện mô hình 2 thành công.")
        print("Huấn luyện mô hình 2 thành công.")
    except Exception as e:
        logging.error(f"Lỗi trong quá trình huấn luyện mô hình 2: {e}")
        print(f"Lỗi trong quá trình huấn luyện mô hình 2: {e}")
def daily_task():
    logging.info("Tác vụ đã được thực thi!")
    print("Tác vụ đã được thực thi!")
    train_model_Light_GCN()
    train_model_convmf()
    if os.path.exists(log_file_path):
        print("File log đã được tạo.")
    else:
        print("File log chưa được tạo.")

schedule.every().day.at("13:43").do(daily_task)

while True:
    schedule.run_pending()
    time.sleep(60)  


