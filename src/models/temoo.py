# coding: utf-8
r"""
TEMOO
"""


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import time




from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian,build_knn_neighbourhood


class TEMOO(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TEMOO, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']
        self.co_topk = config['co_topk']
        self.n_nodes = self.n_users + self.n_items
        self.cl_loss = config['cl_loss']
        
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}_{}.pt'.format(self.knn_k, self.mm_image_weight))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        #Build the item co-occurrence graph
        item_co_graph_file = os.path.join(dataset_path, 'item_co_graph_dict.pt')
        if os.path.exists(item_co_graph_file):
            self.item_co_graph = torch.load(item_co_graph_file)
        else:
            self.item_co_graph = self.get_items_co_graph(self.interaction_matrix)
            torch.save(self.item_co_graph, item_co_graph_file)

        #normalization
        item_co_graph_norm_file = os.path.join(dataset_path, 'item_co_graph_norm_{}.pt'.format(self.co_topk))
        if os.path.exists(item_co_graph_norm_file):
            del self.item_co_graph
            self.item_co_graph_norm = torch.load(item_co_graph_norm_file)
        else:
            self.item_co_graph_norm = self.item_co_graph.to_dense()
            topk_values, topk_indices = self.item_co_graph_norm.topk(self.co_topk, dim=1)
            self.item_co_graph_norm.fill_(0.)
            self.item_co_graph_norm.scatter_(1, topk_indices, topk_values)
            self.item_co_graph_norm = F.normalize(self.item_co_graph_norm, p=1, dim=1)
            self.item_co_graph_norm = self.item_co_graph_norm.to_sparse()
            torch.save(self.item_co_graph_norm, item_co_graph_norm_file)
            del self.item_co_graph

        # Modality Similarity graph
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)


    def get_items_co_graph(self, train_interactions):
        train_interactions = torch.sparse.FloatTensor(
            torch.LongTensor([train_interactions.row, train_interactions.col]),
            torch.FloatTensor(train_interactions.data),
            torch.Size(train_interactions.shape)
        ).to(self.device)
        
        co_occurrence_graph = torch.sparse.mm(train_interactions.t(), train_interactions)

        co_occurrence_graph = co_occurrence_graph.coalesce()
        values = co_occurrence_graph.values()
        indices = co_occurrence_graph.indices()
        mask = indices[0] != indices[1]
        new_indices = indices[:, mask]
        new_values = values[mask]
        co_occurrence_graph = torch.sparse.FloatTensor(new_indices, new_values, co_occurrence_graph.size())
        
        return co_occurrence_graph


    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))



    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            self.R = self.norm_R
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

        
    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):

        hm = self.item_id_embedding.weight
        for i in range(self.n_layers):
            hm = torch.sparse.mm(self.mm_adj, hm)

        hco = self.item_id_embedding.weight
        for i in range(self.n_layers):
            hco = torch.sparse.mm(self.item_co_graph_norm, hco)
        control_item_embeds = hco + hm
        control_embeds = torch.sparse.mm(adj, torch.cat((self.user_embedding.weight.detach(), control_item_embeds), dim=0))
        control_user_embeds,__ = torch.split(control_embeds, [self.n_users, self.n_items], dim=0)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings+control_user_embeds, i_g_embeddings + control_item_embeds, hm, hco, i_g_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)



    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings,mm_embeddings,co_embeddings,i_g_embeddings= self.forward(self.masked_adj)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        cl_loss1 = self.InfoNCE(mm_embeddings[pos_items], co_embeddings[pos_items], 0.2)
        cl_loss2 = self.InfoNCE(i_g_embeddings[pos_items], co_embeddings[pos_items] + mm_embeddings[pos_items], 0.2)
        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,neg_i_g_embeddings)

        return batch_mf_loss  + self.cl_loss * cl_loss1 + self.cl_loss * cl_loss2 

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e ,__,__,__= self.forward(self.norm_adj)
        
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
       
        return scores

