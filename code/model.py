import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import math
from torch.utils.data import DataLoader


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.to(device)
    
    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        try:
            a = self.attn_fc(z2)
        except:
            print(z2, edges.src['z'], edges.dst['z'])
            os.system('pause')
        e = F.leaky_relu(a)
        edges.data['e'] = e
        return {'e': e}
    
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def update_edge_e(self, h, g):
        self.eval()
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.ndata.pop('z')
        return g.edata['e']

    def forward(self, h, g, counter):
        """Return 以最后一个node为root的gcn输出
        :params h: 所有节点emb
        """
        z = self.fc(h)
        g.ndata['z'] = z
        g.ndata['h'] = z
        g.apply_edges(self.edge_attention, edges='__ALL__')
        it = 0
        while counter > it:
            g.update_all(self.message_func, self.reduce_func)
            g.ndata['z'] = g.ndata['h']
            it += 1
        g.ndata.pop('z')
        return g.ndata['h']
		
		
class MyNet(nn.Module):
    def __init__(self, num_entities, num_relation, dim, device, hidden_unit=256, 
            margin=1.0):
        super(MyNet, self).__init__()

        self.num_entity = num_entities
        self.num_relation = num_relation
        self.dim = dim
        
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.gat = GATLayer(dim, dim, device)
        # self.bi_atten = BilinearAttention(dim)
        self.dropout = nn.Dropout(0)
        self.device = device

        self.relation_emb = nn.Embedding(self.num_relation, self.dim * self.dim)
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.to(device)

    def get_preference(self, graph_l, graph_s, l_counter, s_counter, l_length, s_length):
        ndata_emb_l = self.entity_emb(graph_l.ndata['idx']).squeeze(1)
        l_interest_v = self.gat(ndata_emb_l, graph_l, max(l_counter)-1).reshape(-1, self.dim)
        l_unbatch_g = [gg0.ndata['h'] for gg0 in dgl.unbatch(graph_l)]
        l_interest_v = torch.cat([gg0[l_length[index]-1] for index, gg0 in enumerate(l_unbatch_g)], dim=0).view(-1, 1, self.dim)

        ndata_emb_s = self.entity_emb(graph_s.ndata['idx']).squeeze(1)
        s_interest_v = self.gat(ndata_emb_s, graph_s, max(s_counter)-1).reshape(-1, self.dim)
        s_unbatch_g = [gg0.ndata['h'] for gg0 in dgl.unbatch(graph_s)]
        s_interest_v = torch.cat([gg0[s_length[index]-1] for index, gg0 in enumerate(s_unbatch_g)], dim=0).view(-1, 1, self.dim)

        query = torch.cat((l_interest_v, s_interest_v), dim=1)
        return query

    def forward_kg(self, h, r, t, hn, rn, tn):
        kge_loss = 0

        h_emb = self.entity_emb(h)
        t_emb = self.entity_emb(t).transpose(1, 2)
        r_emb = self.relation_emb(r).view(-1, self.dim, self.dim)
        hn_emb = self.entity_emb(hn)
        tn_emb = self.entity_emb(tn).transpose(1, 2)
        rn_emb = self.relation_emb(rn).view(-1, self.dim, self.dim)
        distance1 = torch.squeeze(
            torch.matmul(torch.matmul(h_emb, r_emb), t_emb)
        )
        distance2 = torch.squeeze(
            torch.matmul(torch.matmul(hn_emb, rn_emb), tn_emb)        
        )
        hRt = torch.squeeze(
            torch.matmul(torch.matmul(h_emb, r_emb), t_emb)
        )
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        
        return self.criterion(distance1, distance2, target).mean()

    def forward(self, userId, news_entities_id, graph_l, graph_s, l_counter, s_counter, update_emb, l_length, s_length):
        """Returns the value of the outputs, the predictions of this model
        
        :params userId: user id 
        :params news_entities_id: candidate news entities ids
        :params graph_l(dgl object): long-term interest graph
        :params graph_s(dgl object): short-term interest graph
        """
        ndata_emb_l = self.entity_emb(graph_l.ndata['idx']).squeeze(1)
        l_interest_v = self.gat(ndata_emb_l, graph_l, max(l_counter)-1).reshape(-1, self.dim)

        ndata_emb_list = []

        l_unbatch_g = [gg0.ndata['h'] for gg0 in dgl.unbatch(graph_l)]
        l_interest_v = torch.cat([gg0[l_length[index]-1] for index, gg0 in enumerate(l_unbatch_g)], dim=0).view(-1, 1, self.dim)

        ndata_emb_s = self.entity_emb(graph_s.ndata['idx']).squeeze(1)
        s_interest_v = self.gat(ndata_emb_s, graph_s, max(s_counter)-1).reshape(-1, self.dim)

        ndata_emb_s_list = []

        s_unbatch_g = [gg0.ndata['h'] for gg0 in dgl.unbatch(graph_s)]
        s_interest_v = torch.cat([gg0[s_length[index]-1] for index, gg0 in enumerate(s_unbatch_g)], dim=0).view(-1, 1, self.dim)

        if update_emb:
            ndata_emb_list = l_unbatch_g
            ndata_emb_s_list = s_unbatch_g

        mask = torch.where(news_entities_id==201659, torch.full_like(news_entities_id, 0), news_entities_id).unsqueeze(1).to(torch.float)
        news_v = self.entity_emb(news_entities_id).transpose(1, 2)
        

        query = torch.cat((l_interest_v, s_interest_v), dim=1)
        scores = torch.bmm(query, news_v) / math.sqrt(query.shape[-1])
        atten_weights = self.dropout(F.softmax(scores, dim=1))
        tmp = torch.sum(scores.mul(atten_weights), 1, keepdim=True)
        outs = torch.sum(torch.where(mask==0, mask, tmp), 2, keepdim=True)
        #outs = torch.bmm(scores.transpose(1, 2), atten_weights)
        return outs, ndata_emb_list, ndata_emb_s_list#, edata_emb
        
