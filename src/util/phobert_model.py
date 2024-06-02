from transformers import AutoModel, AutoTokenizer
import re
from pyvi import ViTokenizer as tokenize
import torch

class Phobert:
    __instance = None

    @staticmethod
    def get_instance():
        if Phobert.__instance is None:
            Phobert()
        return Phobert.__instance

    def __init__(self):
        if Phobert.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            self.model = AutoModel.from_pretrained("vinai/phobert-base-v2")

            stopwords_file = open('stop_words_Vietnamese.txt','r')
            content = stopwords_file.read()
            stopwords_VN = content.splitlines()
            stopwords_file.close()

            stopwords_file = open('stop_words_English.txt','r')
            content = stopwords_file.read()
            stopwords_EN = content.splitlines()
            stopwords_file.close()

            self.stopwords = stopwords_VN + stopwords_EN
    
            Phobert.__instance = self

    def get_embedding(self, text):
        text = self.vitokenize(text)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(input_ids)
        return output.pooler_output.squeeze().tolist()
    
    
    def tienxuly(self,document):
        document = document.lower()
        document = re.sub(r'https?://\S+|www\.\S+', ' ', document)
        document = re.sub(r'<.*?>', ' ', document)
        document = re.sub(r'\[.*?\]', ' ', document)
        document = re.sub(r'\n', ' ', document)
        document = re.sub(r'\w*\d\w*', ' ', document)

        document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)
        document = re.sub(r'\s+', ' ', document).strip()
        return document

    def remove_stopwords(self,line):
        words = []
        for word in line.strip().split():
            if word not in self.stopwords:
                words.append(word)
        return ' '.join(words)

    def vitokenize(self,text):
        text = self.tienxuly(text)
        text = self.remove_stopwords(text)
        return tokenize.tokenize(text)
    