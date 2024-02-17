import torch
import torch.nn as nn

class VBPR(nn.Module):
    def __init__(self, n_user, n_item, K, D, img_embedding, emb_norm="None") -> None:
        super().__init__()
        self.K = K                      # general emb dim
        self.D = D                      # visual emb dim
        self.n_user = n_user
        self.n_item = n_item
        self.emb_norm = emb_norm        # emb normalization method, defalut:None
        self.feat_map = img_embedding.float() # item * F
        self.F = self.feat_map.shape[1]       # F = 512

        self.offset = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(self.n_user,1)     # user*1
        self.item_bias = nn.Embedding(self.n_item,1)     # item*1
        self.vis_bias = nn.Embedding(self.F,1)           # F*1
        self.user_emb = nn.Embedding(self.n_user,self.K) # user*K
        self.item_emb = nn.Embedding(self.n_item,self.K) # item*K
        self.item_vis_emb = nn.Embedding(self.F, self.D) # F*D
        self.user_vis_emb = nn.Embedding(self.n_user, self.D) # user*D
        
        if self.emb_norm == "Batch":
            self.user_bn = nn.BatchNorm1d(self.K)
            self.item_bn = nn.BatchNorm1d(self.K)
            self.item_vis_bn = nn.BatchNorm1d(self.D)
            self.user_vis_bn = nn.BatchNorm1d(self.D)
    
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
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        user_vis_emb = self.user_vis_emb(user)
        item_vis_emb = self.item_vis_emb.weight
        
        if self.emb_norm == "L2":
            user_emb = user_emb / torch.linalg.norm(user_emb, dim=1).unsqueeze(dim=1)
            item_emb = item_emb / torch.linalg.norm(item_emb, dim=1).unsqueeze(dim=1)
            user_vis_emb = user_vis_emb / torch.linalg.norm(user_vis_emb, dim=1).unsqueeze(dim=1)
            item_vis_emb = item_vis_emb / torch.linalg.norm(item_vis_emb, dim=1).unsqueeze(dim=1)

        if self.emb_norm == "Batch":
            user_emb = self.user_bn(user_emb)
            item_emb = self.item_bn(item_emb)
            user_vis_emb = self.user_vis_bn(user_vis_emb)
            item_vis_emb = self.item_vis_bn(item_vis_emb)
        
        vis_term = ((user_vis_emb)*(self.feat_map[item]@self.item_vis_emb.weight)).sum(dim=1).unsqueeze(-1) + (self.feat_map[item])@(self.vis_bias.weight)
        mf_term = self.offset + self.user_bias(user) + self.item_bias(item) + (user_emb*item_emb).sum(dim=1).unsqueeze(-1)
        params = (self.offset, self.user_bias(user), self.item_bias(item), self.vis_bias.weight, user_emb, item_emb, item_vis_emb, user_vis_emb)

        return (mf_term+vis_term).squeeze(), params
    
    def forward(self, user, pos, neg):
        xui, pos_params = self.cal_each(user,pos)
        xuj, neg_params = self.cal_each(user,neg)
        return (xui-xuj), pos_params, neg_params

class BPRLoss(nn.Module):
    def __init__(self, visual=True, reg_theta=0, reg_beta=0, reg_e=0) -> None:
        super().__init__()
        self.reg_theta = reg_theta
        self.reg_beta = reg_beta
        self.reg_e = reg_e
        self.visual = visual

    def _cal_l2(self, *tensors):
        total = 0
        for tensor in tensors:
            total += tensor.pow(2).sum()
        return 0.5 * total

    def _reg_term(self, pos_params, neg_params):
        if self.visual:
            # VBPR
            alpha, beta_u, beta_pos, beta_prime_pos, gamma_u, gamma_pos, e_pos, theta_u = pos_params
            _, _, beta_neg, beta_prime_neg, _, gamma_neg, e_neg, _ = neg_params
            reg_out = self.reg_theta * self._cal_l2(alpha, beta_u, beta_pos, beta_neg, theta_u, gamma_u, gamma_pos, gamma_neg)
            reg_out += self.reg_beta * self._cal_l2(beta_prime_pos, beta_prime_neg)
            reg_out += self.reg_e * self._cal_l2(e_pos, e_neg)
        else:
            # BPR
            alpha, beta_u, beta_pos, gamma_u, gamma_pos = pos_params
            _, _, beta_neg, _, gamma_neg = neg_params
            reg_out = self.reg_theta * self._cal_l2(alpha, beta_u, beta_pos, beta_neg, gamma_u, gamma_pos, gamma_neg)

        return reg_out

    def forward(self, diff, pos_params, neg_params):
        loss = -nn.functional.logsigmoid(diff).sum() # logsigma(x_uij)
        loss += self._reg_term(pos_params, neg_params) # reg_term
        return loss
    
class BPRMF(nn.Module):
    def __init__(self, n_user, n_item, K) -> None:
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.K = K

        self.offset = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(self.n_user,1) # user*1
        self.item_bias = nn.Embedding(self.n_item,1) # item*1
        self.user_emb = nn.Embedding(self.n_user,self.K) # user*K
        self.item_emb = nn.Embedding(self.n_item,self.K) # item*K
    
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_bias.weight)
        nn.init.xavier_uniform_(self.item_bias.weight.data)
        nn.init.xavier_uniform_(self.user_emb.weight.data)
        nn.init.xavier_uniform_(self.item_emb.weight.data)
    
    def cal_each(self, user, item):
        mf_term = self.offset + self.user_bias(user) + self.item_bias(item) + (self.user_emb(user)*self.item_emb(item)).sum(dim=1).unsqueeze(-1)
        params = (self.offset, self.user_bias(user), self.item_bias(item), self.user_emb(user), self.item_emb(item))
        return mf_term.squeeze(), params
    
    def forward(self, user, pos, neg):
        xui, pos_params = self.cal_each(user,pos)
        xuj, neg_params = self.cal_each(user,neg)
        return (xui-xuj), pos_params, neg_params