import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from random import uniform, sample
import numpy as np


class TrainDataset():
    """Return userId, news_entities_id, graph_l, graph_s, label
    train_df [label, userid, newsid]
    
    
    """
    def __init__(self, train_df, graph_labels, news2entities):
        self.train_df = train_df
        self.graph_labels = graph_labels
        self.news2entities = news2entities

    def __len__(self):
        return self.train_df.shape[0]

    def __getitem__(self, idx):
        userId, newsId, label = self.train_df.loc[idx, ['userId', 'newsId', 'label']].values
        if not np.argwhere(self.graph_labels['glabel'].numpy() == userId):
            print(userId)
            os.system('pause')
        graph_idx = np.argwhere(self.graph_labels['glabel'].numpy() == userId)[0][0]
        ## suppose long-term = short-term
        # graph_l = self.graph_list[graph_idx]
        # graph_s = self.graph_list[graph_idx]

        ## 使用一个entity代表news测试
        news_entities = self.news2entities[newsId]
        dct = {
            'userId': torch.tensor(int(userId), dtype=torch.float), 
            'news_entities_id': torch.tensor(news_entities, dtype=torch.long),
            'graph_l': graph_idx,
            'graph_s': graph_idx, 
            'label': torch.tensor(int(label), dtype=torch.float)
        }
        return dct

class KGDataset():
    def __init__(self, triple_df, entity2id):
        self.triple_df = triple_df
        self.entity2id = entity2id

    def __len__(self):
        return self.triple_df.shape[0]

    def __getitem__(self, idx):
        h, r, t = self.triple_df.loc[idx, ['h', 'r', 't']].values
        i = uniform(-1, 1)
        if i < 0:
            while True:
                entity_tmp = sample(list(self.entity2id.values()), 1)[0]
                if entity_tmp != h:
                    break
            hn, rn, tn = entity_tmp, r, t
        else:
            while True:
                entity_tmp = sample(list(self.entity2id.values()), 1)[0]
                if entity_tmp != t:
                    break
            hn, rn, tn = h, r, entity_tmp
        dct = {
            'h': torch.LongTensor([h]),
            'r': torch.LongTensor([r]),
            't': torch.LongTensor([t]),
            'hn': torch.LongTensor([hn]), 
            'rn': torch.LongTensor([rn]),
            'tn': torch.LongTensor([tn])
        }
        return dct
