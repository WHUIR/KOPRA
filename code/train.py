import sys
import os
os.chdir('your data path')
import numpy as np
import pandas as pd
import gc
import math
from datetime import datetime
import copy
from random import uniform, sample

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
from dgl.data.utils import save_graphs, load_graphs

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

import pickle
from tqdm import tqdm

from dataloader import TrainDataset, KGDataset
from utils import *
from model import MyNet

## Hyperparams
EPOCHES = 5
INIT_TIMES = 3
BATCH_SIZE = 128
DIM = 20
HIDDEN_UNIT = 8
THRESHOLD = 0.5
NUM_OF_ENTITIES = # number of entities
NUM_OF_RELATIONS = # number of relations

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
STEP_SIZE = 20
GAMMA = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def KG_train_fn(model, optimizer, train_dataloader, device):
    model.train()
    final_loss = 0.
    for data in tqdm(train_dataloader):
        h, r, t, hn, rn, tn = data['h'].to(device), data['r'].to(device), data['t'].to(device), data['hn'].to(device), data['rn'].to(device), data['tn'].to(device)
        loss = model.forward_kg(h, r, t, hn, rn, tn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss += loss.item()

    print('LOSS: {}'.format(final_loss))
    return final_loss

def train_fn(model, graph_list, graph_list_s, dataloader, optimizer, scheduler, loss_fn, update_emb, length_list, length_list_s, device):
    model.train()
    final_loss = 0.
    p_c, n_c = 0, 0
    y_pred = []
    y_label = []
    for data in tqdm(dataloader):

        l_g_batch = dgl.batch([graph_list[m] for m in data['graph_l']])
        l_counter = [len(graph_list[m].nodes()) for m in data['graph_l']]
        l_length = [length_list[m] for m in data['graph_l']]
        s_g_batch = dgl.batch([graph_list_s[m] for m in data['graph_s']])
        s_counter = [len(graph_list_s[m].nodes()) for m in data['graph_s']]
        s_length = [length_list_s[m] for m in data['graph_l']]
        pred, ndata_emb, ndata_emb_s = model(data['userId'].to(device), data['news_entities_id'].to(device), l_g_batch.to(device), s_g_batch.to(device), l_counter, s_counter, update_emb, l_length, s_length)
        # pred, ndata_emb, ndata_emb_s, edata_emb = model(data['userId'].to(device), data['news_entities_id'].to(device), graph_list[data['graph_l']].to(device), graph_list_s[data['graph_s']].to(device))
        if update_emb:
            for idx, m in enumerate(data['graph_l']):
                graph_list[m].ndata['emb'] = ndata_emb[idx].to('cpu')
            for idx, m in enumerate(data['graph_s']):
                graph_list_s[m].ndata['emb'] = ndata_emb_s[idx].to('cpu')

        target = data['label'].view(-1, 1).to(device)
        for res0 in range(target.shape[0]):
            if (target[res0] == 1 and pred[res0] > 0) or (target[res0] == 0 and pred[res0] <= 0):
                p_c += 1
            else:
                n_c += 1

        # print(pred, target)
        y_pred.extend((pred.view(-1).detach().cpu().numpy()+10)/20)
        y_label.extend(target.view(-1).detach().cpu().numpy())
        loss = loss_fn(pred.squeeze(1), target)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        final_loss += loss.item()
    print('ACC: {}, AUC: {}'.format(p_c / (p_c + n_c), roc_auc_score(y_label, y_pred)))
    final_loss /= len(dataloader)
    return final_loss


def valid_fn(model, graph_list, graph_list_s, dataloader, length_list, length_list_s, device):
    model.eval()
    p_c, n_c = 0, 0
    y_pred = []
    y_label = []
	users = []
    print('Get user preference...')
    users_pre = None

    for data in tqdm(dataloader):
        l_g_batch = dgl.batch([graph_list[m] for m in data['graph_l']])
        l_counter = [len(graph_list[m].nodes()) for m in data['graph_l']]
        l_length = [length_list[m] for m in data['graph_l']]
        s_g_batch = dgl.batch([graph_list_s[m] for m in data['graph_s']])
        s_counter = [len(graph_list_s[m].nodes()) for m in data['graph_s']]
        s_length = [length_list_s[m] for m in data['graph_l']]
        pred, ndata_emb, ndata_emb_s = model(data['userId'].to(device), data['news_entities_id'].to(device), l_g_batch.to(device), s_g_batch.to(device), l_counter, s_counter, False, l_length, s_length)
        target = data['label'].view(-1, 1).to(device)
        for res0 in range(target.shape[0]):
            if (target[res0] == 1 and pred[res0] > 0) or (target[res0] == 0 and pred[res0] <= 0):
                p_c += 1
            else:
                n_c += 1
            y_pred.extend((pred.view(-1).detach().cpu().numpy() + 10) / 20)
            y_label.extend(target.view(-1).detach().cpu().numpy())
			users.extend(data['userId'])
	res = calculate_group_metric(y_label, y_pred, users)
    print('ACC: {}, AUC: {}'.format(p_c / (p_c + n_c), roc_auc_score(y_label, y_pred)))
    return

def train(train_df, test_df, kg_df, graph_list, graph_list_s, graph_labels, news2entities, entity2id, kg, device, is_init=False):

	length_list = [len(g.nodes()) for g in graph_list_s]
    length_list_s = [len(g.nodes()) for g in graph_list_s]
    
    model = MyNet(NUM_OF_ENTITIES, NUM_OF_RELATIONS, DIM, DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    loss_fn = nn.BCEWithLogitsLoss()
    
    train_dataset = TrainDataset(train_df, graph_labels, news2entities)
    valid_dataset = TrainDataset(test_df, graph_labels, news2entities)
    kg_dataset = KGDataset(kg_df, entity2id)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    kg_dataloader = DataLoader(kg_dataset, batch_size=BATCH_SIZE * 2, shuffle=True)
    ## load checkpoint
	if is_init:
		checkpoint = torch.load('##checkpoint_path##')
		model.load_state_dict(checkpoint['net'])
		optimizer.load_state_dict(checkpoint['optimizer'])

	## 初始化的时候不prume
    if INIT_TIMES > 0:
		## init embedding
		train_loss = train_fn(model, graph_list, graph_list_s, train_dataloader, optimizer, scheduler, loss_fn, True, length_list, length_list_s, device)
		print('Init embedding : Train loss: {}'.format(train_loss))
		# valid_fn(model, graph_list, graph_list_s, valid_dataloader, length_list, length_list_s, device)
		INIT_TIMES -= 1

    
    for i_idx in range(30):
        ## 对每次剪枝重复epoches遍
        for e in range(EPOCHES):
            ## move to cpu
            model.to('cpu')
            if i_idx > 0 and (e == 0 or e == EPOCHES-1):
                ## init edata[e]
                ## update edge.data['e']
                print('Interest: {}, Epoch: {}, init edge data e...'.format(i_idx, e))
                for idx in tqdm(range(len(graph_list))):
                    # print(g0.edata['e'])
                    if len(graph_list[idx].nodes()) == 1:
                        continue
                    model.eval()
                    h = graph_list[idx].ndata['emb']
                    edata_e = model.gat.update_edge_e(h, graph_list[idx])
                    graph_list[idx].edata['e'] = edata_e
                    
                ## prume
                # 遍历所有的用户图
                print('process each user graph...')
                for idx, g0 in enumerate(tqdm(graph_list)):
                    # print('prume graph {}, nodes: {}'.format(idx, len(g0.nodes())))
                    # 判断g0的长度
                    if i_idx > length_list[idx]:
                        continue
                    
                    # kg扩展seed (seed, memory_size, kg, hop_num=1)
                    subgraph = build_subgraph_set(g0.nodes[i_idx-1].data['idx'], 10, kg, hop_num=1)
                    
                    preference_v = gcn_from_tree(g0, i_idx-1, stem_list[idx], model)
                    prumed_g = prume_fn(subgraph, preference_v, model, THRESHOLD, i_idx-1)
                    # prumed_g [(t, node_idx)]
                    
                    updated_emb = g0.ndata['emb']
                    updated_idx = g0.ndata['idx']
                    for (t, node_idx) in prumed_g:
                        total_nodes = g0.number_of_nodes()
                        g0.add_edges(total_nodes, node_idx)
                        g0.add_edges(node_idx, total_nodes)
                        tmp_emb = model.eval().entity_emb(torch.LongTensor([t]))
                        updated_emb = torch.cat((updated_emb, tmp_emb))
                        tmp_idx = torch.LongTensor([t]).view(-1, 1)
                        updated_idx = torch.cat((updated_idx, tmp_idx))

                    # print(updated_emb.shape, g0.ndata['emb'].shape)
                    g0.ndata['emb'] = updated_emb
                    g0.ndata['idx'] = updated_idx
                    try:
                        g0.ndata.pop('_ID')
                        g0.edata.pop('_ID')
                    except:
                        pass
                    
                for idx in range(len(graph_list)):
                    try:
                        graph_list[idx].edata.pop('e')
                    except:
                        pass
                
            ## move to GPU
            model.to(device)    
            
            ## model, graph_list, dataloader, optimizer, scheduler, loss_fn, device
            update_emb = False
            if e == EPOCHES - 1:
                update_emb = True

            kg_loss = KG_train_fn(model, optimizer, kg_dataloader, device)
            #print('Interest: {}, Epoch: {}, KG_loss: {}'.format(i_idx, e, kg_loss))
            train_loss = train_fn(model, graph_list, graph_list_s, train_dataloader, optimizer, scheduler, loss_fn, update_emb, length_list, length_list_s, device)
            print('Interest: {}, Epoch: {}, Train loss: {}'.format(i_idx, e, train_loss))
            valid_fn(model, graph_list, graph_list_s, valid_dataloader, length_list, length_list_s, device)
            ## saving
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': e}
            torch.save(state, '##checkpoint_path##')

    save_graphs('./model/pruned_l_graph.bin', graph_list, graph_labels)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    print('Reading data...')
    train0 = pd.read_csv('##train_file##.csv', sep=' ', index_col=[0])
	test_df = pd.read_csv('##train_file##.csv', sep=' ', index_col=[0])

    news_entity_dct = pickle.load(open('##news2entities##.pkl', 'rb'))

    ## get graph_list, graph_labels
    graph_list, graph_labels = load_graphs('##long_term_graphs##.bin')

    graph_list_s, _ = load_graphs('##short_term_graphs##.bin')
    stem_list = [set([i for i in range(len(g.nodes()))]) for g in graph_list]
    
	## KG
    kg = pickle.load(open('##kg_to_add_nodes##.pkl', 'rb'))
    kg_df = pd.read_csv('##kg_to_trainKg##.csv', sep='\t')
    entity2id = pickle.load(open('##entity2id##.pkl', 'rb'))
    # train
    train(train_df, test_df, kg_df, graph_list, graph_list_s, graph_labels, news_entity_dct, entity2id, kg, DEVICE, is_init=False)
