import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import DataLoader

import dgl


def edge_attention(edges):
    e = edges.data['e']
    return {'e': e}

def edge_attention_without_e(edges):
    z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    a = torch.mean(z2, dim=1) 
    e = F.leaky_relu(a)
    edges.data['e'] = e
    return {'e': e}


def message_func(edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

def reduce_func(nodes):
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    try:
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    except:
        print(alpha.shape, nodes.mailbox['z'].shape)
        os.system('pause')
    ## 缺少正则化
    return {'h': h}

def reverse_g(graph, stem_set):
    """
    :param graph: 用户兴趣graph
    :param stem_set: seed set 存储seeds
    """
    for (src, dic) in zip(graph.edges()[0].tolist(), graph.edges()[1].tolist()):
        if src in stem_set and dic in stem_set:
            ids = graph.edge_ids(src, dic)
            graph.remove_edges(torch.tensor([ids], dtype=torch.int64))
            graph.add_edge(dic, src)
    

def gcn_from_tree(g, node_idx, stem_set, model):
    if len(g.nodes()) == 1:
        return g.ndata['emb']
    features = g.ndata['emb']
    
    with torch.no_grad():
        inputs = model.gat.fc(features)
        g.ndata['z'] = inputs

        # if 'e' in g.edata:
        #     g.apply_edges(model.gat.edge_attention)
        # else:
        #     g.apply_edges(edge_attention_without_e)
        g.apply_edges(edge_attention)
        times = node_idx
        while times >= 0:
            g.update_all(message_func, reduce_func)
            g.ndata['z'] = g.ndata['h']
            times -= 1
        g.ndata.pop('z')
        sub = g.ndata.pop('h')[node_idx]
        return sub
		
class BilinearAttention(nn.Module):
    def __init__(self, dim):
        super(BilinearAttention, self).__init__()
        self.fc = nn.Linear(dim, dim, bias=False)

    def forward(self, x, q):
        ## [batch_size, num_domain, dim]
        k_m = self.fc(x) 
        ## x [batch_size, num_domain, dim]
        ## q [batch_size, dim]
        q = q.transpose(1, 2)
        outs = torch.bmm(k_m, q).squeeze(2)
        return F.softmax(outs)  
		
def DotProductSimilarity(t_v, p_v, scale_output=False):
    score = (t_v * p_v).sum(dim=-1)
    if scale_output:
        score /= math.sqrt(t_v.shape[-1])
    return score  
		
def build_subgraph_set(seed, memory_size, kg, hop_num=1):
    """Return the subgraph_set
    [(h, t), (h, t)...]
    
    :params seed_set: user's click hist. :list:
    :params memory_size: the max num of entities in a subgraph :int:
    :params hop_num: the max distance we consider :int: 
    :params kg: knowledge graph
    """
    ## seed = 1 ## entityId

    subgraph0 = []
    edge_list = {}
    for h in range(hop_num):
        counter = 0
        try:
            candidate_list = kg[seed.item()][: memory_size]
            #print('found {}'.format(seed.item()))
        except:
            ## 没有seed可扩展
            # print('entity {} does not have tails'.format(seed.item()))
            break
        edge_list[h] = []
        for item in candidate_list:
            edge_list[h].append((int(item), seed))
        
    for edges in edge_list.values():
        subgraph0 += edges
    
    return subgraph0
	
def prume_fn(subgraph, preference_v, model, threshold, node_index):
    """Return prumed subgraph
    :param subgraph: 经过扩展的子图 ([(h, t), (h, t)...])
    :param preference_v: 用户preference representation
    :param model: trained emb
    :param threshold: 阈值
    :param node_idx: user graph node idx
    """
    res = []
    for idx, (h, t) in enumerate(subgraph):
        with torch.no_grad():
            t_v = model.entity_emb(torch.LongTensor([h]))
            score = DotProductSimilarity(t_v, preference_v)
            if -(threshold) < score < threshold:
                pass
            else:
                res.append((h, node_index))
    return res

def prume_fn_wo(subgraph, node_index):
    res = []
    for idx, (h, t) in enumerate(subgraph):    
        res.append((h, node_index))
    return res
	
	
def calculate_group_metric(labels, preds, users, calc_ndcg=True, calc_hit=True, calc_mrr=True,
                           at_Ns=None):
    if at_Ns is None:
        at_Ns = [##***##] # @number
    metrics = {}

    user_pred_dict = {}

    print_time_cost = False

    for i in range(len(users)):
        if users[i] in user_pred_dict:
            user_pred_dict[users[i]][0].append(preds[i])
            user_pred_dict[users[i]][1].append(labels[i])
        else:
            user_pred_dict[users[i]] = [[preds[i]], [labels[i]]]

    t = time.time()
    if calc_ndcg or calc_hit or calc_mrr:
        for user, val in user_pred_dict.items():
            idx = np.argsort(val[0])[::-1]
            user_pred_dict[user][0] = np.array(val[0])[idx]
            user_pred_dict[user][1] = np.array(val[1])[idx]

    if calc_ndcg or calc_hit or calc_mrr:
        ndcg = np.zeros(len(at_Ns))
        hit = np.zeros(len(at_Ns))
        mrr = np.zeros(len(at_Ns))
        valid_user = 0
        for u in user_pred_dict:
            if 1 in user_pred_dict[u][1] and 0 in user_pred_dict[u][1]:  # contains both labels
                valid_user += 1
                pred = user_pred_dict[u][1]
                rank = np.nonzero(pred)[0][0]
                #print(pred, rank)
                for idx, n in enumerate(at_Ns):
                    if rank < n:
                        ndcg[idx] += 1 / np.log2(rank + 2)
                        hit[idx] += 1
                        mrr[idx] += 1 / (rank + 1)
        ndcg = ndcg / valid_user
        hit = hit / valid_user
        mrr = mrr / valid_user
        metrics['ndcg'] = ndcg
        metrics['hit'] = hit
        metrics['mrr'] = mrr
        if print_time_cost:
            print("NDCG TIME: %.4fs" % (time.time() - t))
    return metrics
