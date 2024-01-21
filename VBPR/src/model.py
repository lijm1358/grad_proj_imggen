import torch
import torch.nn as nn

class VBPR(nn.Module):
    def __init__(self, n_user, n_item, K, D, img_embedding) -> None:
        super().__init__()
        self.feat_map= img_embedding.float() # user * 512
        self.n_user = n_user
        self.n_item = n_item
        self.K = K
        self.D = D
        self.F = self.feat_map.shape[1] 

        self.offset = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(self.n_user,1) # user*1
        self.item_bias = nn.Embedding(self.n_item,1) # item*1
        self.vis_bias = nn.Embedding(self.F,1)       # 512*1
        self.user_emb = nn.Embedding(self.n_user,self.K) # user*K
        self.item_emb = nn.Embedding(self.n_item,self.K) # item*K
        self.item_vis_emb = nn.Embedding(self.D, self.F) # D*K
        self.user_vis_emb = nn.Embedding(self.n_user, self.D) # user*D
    
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_bias.weight)
        nn.init.xavier_uniform_(self.item_bias.weight.data)
        nn.init.xavier_uniform_(self.vis_bias.weight.data)
        nn.init.xavier_uniform_(self.user_emb.weight.data)
        nn.init.xavier_uniform_(self.item_emb.weight.data)
        nn.init.xavier_uniform_(self.item_vis_emb.weight.data)
        nn.init.xavier_uniform_(self.user_vis_emb.weight.data)
    
    def cal_each(self, user, item):
        vis_term = ((self.user_vis_emb(user)).matmul(self.item_vis_emb.weight@(self.feat_map[item].T))).sum(dim=1) + (self.vis_bias.weight.T)@(self.feat_map[item].T)
        mf_term = self.offset + self.user_bias(user).T + self.item_bias(item).T + ((self.user_emb(user)).matmul(self.item_emb(item).T)).sum(dim=1).unsqueeze(dim=0)
        params = (self.offset, self.user_bias(user), self.item_bias(item), self.vis_bias.weight, self.user_emb(user), self.item_emb(item), self.item_vis_emb.weight, self.user_vis_emb(user))
        return (mf_term+vis_term).squeeze(), params
    
    def forward(self, user, pos, neg):
        xui, pos_params = self.cal_each(user,pos)
        xuj, neg_params = self.cal_each(user,neg)
        return (xui-xuj), pos_params, neg_params

class BPRLoss(nn.Module):
    def __init__(self, reg_theta, reg_beta, reg_e) -> None:
        super().__init__()
        self.reg_theta = reg_theta
        self.reg_beta = reg_beta
        self.reg_e = reg_e

    def _cal_l2(self, *tensors):
        total = 0
        for tensor in tensors:
            total += tensor.pow(2).sum()
        return 0.5 * total

    def _reg_term(self, pos_params, neg_params):
        alpha, beta_u, beta_pos, beta_prime_pos, gamma_u, gamma_pos, e_pos, theta_u = pos_params
        _, _, beta_neg, beta_prime_neg, _, gamma_neg, e_neg, _ = neg_params

        reg_out = self.reg_theta * self._cal_l2(alpha, beta_u, beta_pos, beta_neg, theta_u, gamma_u, gamma_pos, gamma_neg)
        reg_out += self.reg_beta * self._cal_l2(beta_prime_pos, beta_prime_neg)
        reg_out += self.reg_e * self._cal_l2(e_pos, e_neg)

        return reg_out

    def forward(self, diff, pos_params, neg_params):
        loss = -nn.functional.logsigmoid(diff).sum() # sigma(x_uij)
        loss += self._reg_term(pos_params, neg_params) # reg_term

        return loss
    
class MF(nn.Module):
    def __init__(self, n_user: int, n_item:int, n_factor:int):
        super().__init__()
        self.P = nn.Embedding(n_user, n_factor) # user x factor
        self.Q = nn.Embedding(n_item, n_factor) # itme x factor
        self.user_bias = nn.Embedding(n_user, 1) # user x 1
        self.item_bias = nn.Embedding(n_item, 1) # item x 1

    def forward(self, user_id: torch.tensor, item_id: torch.tensor):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id) 
        b_i = self.item_bias(item_id)

        out = torch.sum(P_u*Q_i, axis=1) + torch.squeeze(b_u) + torch.squeeze(b_i)
        return out.view(-1)