import torch
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor, nn, optim
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import structured_negative_sampling
from torch_sparse import SparseTensor, matmul

def train_model_LGCN():
    # # Đọc file .txt
    rating_df = pd.read_csv('data.txt', delimiter='\t', header=None, names=['userId', 'bookId', 'rating'], index_col='userId')

    user_mapping = {idx: i for i, idx in enumerate(rating_df.index.unique())}
    book_mapping = {idx: i for i, idx in enumerate(rating_df['bookId'].unique())}

    num_users = len(rating_df.index.unique())
    num_books = len(rating_df['bookId'].unique())

    rating_df = pd.read_csv('data.txt', delimiter='\t', header=None, names=['userId', 'bookId', 'rating'])


    edge_index = None
    users = [user_mapping[idx] for idx in rating_df['userId']]
    books = [book_mapping[idx] for idx in rating_df['bookId']]
    # filter for edges with a high rating
    ratings = rating_df['rating'].values
    recommend_bool = torch.from_numpy(ratings).view(-1, 1).to(torch.long) >= 3
    edge_index = [[],[]]
    for i in range(recommend_bool.shape[0]):
        if recommend_bool[i]:
            edge_index[0].append(users[i])
            edge_index[1].append(books[i])

    edge_index = torch.tensor(edge_index)


    # split the edges into train set and test set
    num_ratings = edge_index.shape[1]
    rating_indices = np.arange(num_ratings)

    indices_train, indices_val_test = train_test_split(rating_indices, test_size = 0.2, random_state = 42)
    indices_val, indices_test = train_test_split(indices_val_test, test_size = 0.5, random_state = 42)

    def generate_edge(edge_indices):
        sub_edge_index = edge_index[:, edge_indices]
        num_nodes = num_users + num_books
        edge_index_sparse = SparseTensor(row = sub_edge_index[0],
                                        col = sub_edge_index[1],
                                        sparse_sizes = (num_nodes, num_nodes))
        return sub_edge_index, edge_index_sparse

    train_edge_index, train_sparse_edge_index = generate_edge(indices_train)
    val_edge_index, val_sparse_edge_index = generate_edge(indices_val)
    test_edge_index, test_sparse_edge_index = generate_edge(indices_test)
    edges = structured_negative_sampling(train_edge_index)
    edges = torch.stack(edges, dim=0)
    edges

    def mini_batch_sample(batch_size, edge_index):
    
        edges = structured_negative_sampling(edge_index)
        edges = torch.stack(edges, dim=0)
        indices = torch.randperm(edges.shape[1])[:batch_size]
        batch = edges[:, indices]
        user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
        return user_indices, pos_item_indices, neg_item_indices
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

            users_emb, items_emb = \
            torch.split(x_final, [self.num_users, self.num_items])

            return users_emb, self.users_emb.weight, items_emb, self.items_emb.weight

        def message(self, x):
            return x

        def propagate(self, edge_index, x):
            x = self.message_and_aggregate(edge_index, x)
            return x

        def message_and_aggregate(self, edge_index, x):
            return matmul(edge_index, x)

    def bpr_loss(users_emb, user_emb_0, pos_emb, pos_emb_0, neg_emb, neg_emb_0, lambda_val):
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        losses = -torch.log(torch.sigmoid(pos_scores - neg_scores))
        loss = torch.mean(losses) + lambda_val * \
        (torch.norm(users_emb_0) + torch.norm(pos_emb_0) + torch.norm(neg_emb_0))

        return loss


    def evaluation(model, edge_index, sparse_edge_index, mask_index, k, lambda_val):
        users_emb, users_emb_0, items_emb, items_emb_0 = model.forward(sparse_edge_index)
        edges = structured_negative_sampling(edge_index, contains_neg_self_loops=False)

        user_indices, pos_indices, neg_indices = edges[0], edges[1], edges[2]
        users_emb, users_emb_0 = users_emb[user_indices], users_emb_0[user_indices]
        pos_emb, pos_emb_0 = items_emb[pos_indices], items_emb_0[pos_indices]
        neg_emb, neg_emb_0 = items_emb[neg_indices], items_emb_0[neg_indices]

        loss = bpr_loss(users_emb, users_emb_0, pos_emb, pos_emb_0,
                        neg_emb, neg_emb_0, lambda_val).item()

        users_emb_w = model.users_emb.weight
        items_emb_w = model.items_emb.weight
        rating = torch.matmul(users_emb_w, items_emb_w.T)

        for index in mask_index:
            user_pos_items = get_positive_items(index)
            masked_users = []
            masked_items = []
            for user, items in user_pos_items.items():
                masked_users.extend([user] * len(items))
                masked_items.extend(items)

            rating[masked_users, masked_items] = float("-inf")

        _, top_K_items = torch.topk(rating, k=k)
        users = edge_index[0].unique()
        test_user_pos_items = get_positive_items(edge_index)

        actual_r = [test_user_pos_items[user.item()] for user in users]
        pred_r = []

        for user in users:
            items = test_user_pos_items[user.item()]
            label = list(map(lambda x: x in items, top_K_items[user]))
            pred_r.append(label)

        pred_r = torch.Tensor(np.array(pred_r).astype('float'))


        correct_count = torch.sum(pred_r, dim=-1)
        liked_count = torch.Tensor([len(actual_r[i]) for i in range(len(actual_r))])

        recall = torch.mean(correct_count / liked_count)
        precision = torch.mean(correct_count) / k
        return loss, recall, precision

    def get_positive_items(edge_index):
        pos_items = {}
        for i in range(edge_index.shape[1]):
            user = edge_index[0][i].item()
            item = edge_index[1][i].item()
            if user not in pos_items:
                pos_items[user] = []
            pos_items[user].append(item)
        return pos_items

    def recallAtK(actual_r, pred_r, k):
        correct_count = torch.sum(pred_r, dim=-1)
        # number of items liked by each user in the test set
        liked_count = torch.Tensor([len(actual_r[i]) for i in range(len(actual_r))])

        recall = torch.mean(correct_count / liked_count)
        precision = torch.mean(correct_count) / k

        return recall.item(), precision.item()

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

    # setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LightGCN(num_users, num_books, config['hidden_dim'], config['num_layer'])
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])

    edge_index = edge_index.to(device)
    train_edge_index = train_edge_index.to(device)
    train_sparse_edge_index = train_sparse_edge_index.to(device)

    val_edge_index = val_edge_index.to(device)
    val_sparse_edge_index = val_sparse_edge_index.to(device)

    # training loop
    train_losses = []
    val_losses = []

    for epoch in range(config['num_epoch']):
        for iter in range(config['epoch_size']):
            # forward propagation
            users_emb, users_emb_0, items_emb, items_emb_0 = \
                model.forward(train_sparse_edge_index)

            # mini batching
            user_indices, pos_indices, neg_indices = \
                mini_batch_sample(config['batch_size'], train_edge_index)

            user_indices = user_indices.to(device)
            pos_indices = pos_indices.to(device)
            neg_indices = neg_indices.to(device)

            users_emb, users_emb_0 = users_emb[user_indices], users_emb_0[user_indices]
            pos_emb, pos_emb_0 = items_emb[pos_indices], items_emb_0[pos_indices]
            neg_emb, neg_emb_0 = items_emb[neg_indices], items_emb_0[neg_indices]

            # loss computation
            loss = bpr_loss(users_emb, users_emb_0,
                            pos_emb, pos_emb_0,
                            neg_emb, neg_emb_0,
                            config['lambda'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, recall, precision = evaluation(model, val_edge_index,
                                                val_sparse_edge_index,
                                                [train_edge_index],
                                                config['topK'],
                                                config['lambda'])


        # print('Epoch {:d}: train_loss: {:.4f}, val_loss: {:.4f}, recall: {:.4f}, precision: {:.4f}'\
        #         .format(epoch, loss, val_loss, recall, precision))
        pre = epoch / config['num_epoch'] * 100
        pre = round(pre, 2)
        logging.info(f"Tiến độ huấn luyện mô hình LightGCN: {per}%")  
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        scheduler.step()

    torch.save({
        'model_state_dict': model.state_dict(),
    }, 'trained_model.pth')

    torch.save({
        'model_state_dict': model.state_dict(),
        'num_users': num_users,
        'num_books': num_books,
        'user_mapping': user_mapping,
        'book_mapping': book_mapping,
        'edge_index': edge_index,
    }, 'model_and_mappings.pth')
