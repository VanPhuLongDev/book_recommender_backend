import torch
import os
from torch_geometric.nn.conv import MessagePassing
from torch import nn

import warnings
from glob import glob
from tqdm.auto import trange

# Define LightGCN class
class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, hidden_dim, num_layers):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.users_emb = nn.Embedding(self.num_users, self.hidden_dim)
        self.items_emb = nn.Embedding(self.num_items, self.hidden_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index):
        edge_index_norm = gcn_norm(edge_index, False)

        # The first layer, concat embeddings
        x0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        xs = [x0]
        xi = x0

        # pass x to the next layer
        for i in range(self.num_layers):
            xi = self.propagate(edge_index_norm, x=xi)
            xs.append(xi)

        xs = torch.stack(xs, dim=1)
        x_final = torch.mean(xs, dim=1)

        users_emb, items_emb = torch.split(x_final, [self.num_users, self.num_items])

        return users_emb, self.users_emb.weight, items_emb, self.items_emb.weight

    def message(self, x):
        return x

    def propagate(self, edge_index, x):
        x = self.message_and_aggregate(edge_index, x)
        return x

    def message_and_aggregate(self, edge_index, x):
        return matmul(edge_index, x)

# Configuration dictionary
config = {
    'batch_size': 256,
    'num_epoch': 30,
    'epoch_size': 200,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'topK': 20,
    'lambda': 1e-6,
    'hidden_dim': 32,
    'num_layer': 3,
}

# Define LightGCNRecommender class
class LightGCNRecommender:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_path = os.path.dirname(__file__)
        full_path = os.path.join(base_path, model_path)
        checkpoint = torch.load(full_path)
        # Load cấu trúc model
        self.model = LightGCN(checkpoint['num_users'], checkpoint['num_books'], config['hidden_dim'], config['num_layer'])

        # Load trạng thái model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load mapping
        self.user_mapping = checkpoint['user_mapping']
        self.book_mapping = checkpoint['book_mapping']
        self.reverse_book_mapping = {v: k for k, v in self.book_mapping.items()}
        self.edge_index = checkpoint.get('edge_index', None)
    
    def get_positive_items(self, edge_index):
        pos_items = {}
        for i in range(edge_index.shape[1]):
            user = edge_index[0][i].item()
            item = edge_index[1][i].item()
            if user not in pos_items:
                pos_items[user] = []
            pos_items[user].append(item)
        return pos_items

    def predict(self, user_id, topK):
        self.model.eval()
        pos_items = self.get_positive_items(self.edge_index)
        user = self.user_mapping[user_id]
        user_emb = self.model.users_emb.weight[user]
        scores = self.model.items_emb.weight @ user_emb

        values, indices = torch.topk(scores, k=len(pos_items[user]) + topK)

        movies = [index.cpu().item() for index in indices if index in pos_items[user]]
        topk_movies = movies[:topK]
        book_ids = [list(self.book_mapping.keys())[list(self.book_mapping.values()).index(book)] for book in topk_movies]

        books = [index.cpu().item() for index in indices if index not in pos_items[user]]
        topk_books = books[:topK]
        book_ids = [list(self.book_mapping.keys())[list(self.book_mapping.values()).index(book)] for book in topk_books]
        return book_ids

# Instantiate the LightGCNRecommender
recommender = LightGCNRecommender("model_and_mappings.pth")

# Define the predictTopK function
def predictTopK(user_id, topK):
    recommended_books = None
    try:
        recommended_books = recommender.predict(user_id, topK)
    except Exception as e:
        print (e)
    return recommended_books

    


