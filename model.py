import torch
import torch.nn as nn
import torch.nn.functional as F

class SetTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer1 = nn.Linear(embed_dim, embed_dim)
        self.layer2 = nn.Linear(embed_dim, embed_dim)

        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x):
        num_points = x.size(0)
        x = x.transpose(0, 1)
        x = x.reshape(-1, self.embed_dim)
        x = F.relu(self.layer1(x))
        x = self.layer2(x) 

        x = x.reshape(-1, num_points, self.embed_dim)

        x = x.transpose(0, 1)
        attention_output, _ = self.attention(x, x, x) 
        attention_output = attention_output.transpose(0, 1) 

        output = attention_output.mean(dim=1) 

        return output

class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        attention = F.softmax(self.layer2(layer1_act), dim=0)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, embedding_range, query_name_dict, gamma):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.query_name_dict = query_name_dict

        # margin the distance
        self.gamma = gamma
        
        self.embedding_dim = hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.embedding_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.embedding_dim))

        nn.init.uniform_(tensor=self.entity_embedding, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_embedding, a=-embedding_range, b=embedding_range)

        self.inter = CenterIntersection(self.embedding_dim)

    def cal_distance(self, entity_embedding, query_embedding):
        margined_dist = self.gamma - torch.norm(entity_embedding - query_embedding, p=1, dim=-1)
        return margined_dist

    def embed_query(self, queries, query_structure, idx):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                embedding += r_embedding
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.inter(torch.stack(embedding_list))

        return embedding, idx

    def forward(self, positive_sample, negative_sample, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        for query_structure in batch_queries_dict:
            center_embedding, _ = self.embed_query(batch_queries_dict[query_structure], query_structure, 0)
            all_center_embeddings.append(center_embedding)
            all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)

        if type(positive_sample) != type(None):
            assert len(all_center_embeddings) > 0

            positive_sample_regular = positive_sample[all_idxs]
            positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
            positive_logit = self.cal_distance(positive_embedding, all_center_embeddings)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            assert len(all_center_embeddings) > 0

            negative_sample_regular = negative_sample[all_idxs]
            batch_size, negative_size = negative_sample_regular.shape
            negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
            negative_logit = self.cal_distance(negative_embedding, all_center_embeddings)
        
        else:
            negative_logit = None

        return positive_logit, negative_logit, all_idxs