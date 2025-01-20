# coding=utf-8

import torch
import torch.nn.functional as F
from models.BaseModel import BaseModel
from utils import utils
import numpy as np
from utils import global_p
import torch.nn as nn

class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=1) # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=1)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=1)

        return alpha_embedding, beta_embedding

class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

class Negation(nn.Module):
    def __init__(self):
        super(Negation, self).__init__()

    def neg_feature(self, feature):
        feature = feature
        return feature

    def forward(self, logic):
        logic = 1. / logic
        return logic

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class Negation(nn.Module):
    def __init__(self):
        super(Negation, self).__init__()

    def neg_feature(self, feature):
        feature = feature
        return feature

    def forward(self, logic):
        logic = 1. / logic
        return logic

class NCR(BaseModel):
    append_id = True
    include_id = False
    include_user_features = False
    include_item_features = False

    @staticmethod
    def parse_model_args(parser, model_name='NCR'):
        parser.add_argument('--u_vector_size', type=int, default=64,
                            help='Size of user vectors.')
        parser.add_argument('--i_vector_size', type=int, default=64,
                            help='Size of item vectors.')
        parser.add_argument('--r_weight', type=float, default=10,
                            help='Weight of logic regularizer loss')
        parser.add_argument('--ppl_weight', type=float, default=0,
                            help='Weight of uv interaction prediction loss')
        parser.add_argument('--pos_weight', type=float, default=0,
                            help='Weight of positive purchase loss')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, label_min, label_max, feature_num, user_num, item_num, u_vector_size, i_vector_size,
                 r_weight, ppl_weight, pos_weight, random_seed, model_path,number_interest,number_iter):
        self.u_vector_size, self.i_vector_size = u_vector_size, i_vector_size
        assert self.u_vector_size == self.i_vector_size
        self.ui_vector_size = self.u_vector_size
        self.user_num = user_num
        self.item_num = item_num
        self.r_weight = r_weight
        self.ppl_weight = ppl_weight
        self.pos_weight = pos_weight
        self.sim_scale = 10
        # -------------------MI_extract-------------------------
        self.K = number_interest #number of interest
        self.R = number_iter #number of iteration
        self.logic_gamma = torch.tensor(0.)
        self.logic_weight = r_weight
        # ------------------------------------------------------
        BaseModel.__init__(self, label_min=label_min, label_max=label_max,
                           feature_num=feature_num, random_seed=random_seed,
                           model_path=model_path)

    def _init_weights(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.true = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 0.1, size=self.ui_vector_size).astype(np.float32)), requires_grad=False)
        self.not_layer_1 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.not_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.and_layer_1 = torch.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.and_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.or_layer_1 = torch.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.or_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.purchase_layer_1 = torch.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.purchase_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        # self.output_1 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size, bias=False)
        # self.output_2 = torch.nn.Linear(self.ui_vector_size, 1, bias=False)
        # -------------------MI_extract-------------------------
        # one S for all routing operations, first dim is for batch broadcasting
        S = torch.empty(self.ui_vector_size, self.ui_vector_size)
        torch.nn.init.normal_(S, mean=0.0, std=1.0)
        self.S = torch.nn.Parameter(S) # don't forget to make S as model parameter
        self.dense1 = torch.nn.Linear(self.ui_vector_size, 4 * self.ui_vector_size)
        self.dense2 = torch.nn.Linear(4 * self.ui_vector_size, self.ui_vector_size)
        self.dense3 = torch.nn.Linear(4 * self.ui_vector_size, self.K)
        base_B = torch.nn.init.normal_(torch.empty(self.K, self.ui_vector_size), mean=0.0, std=1.0).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.base_B = torch.nn.Parameter(base_B)
        base_position_embedding = torch.empty(self.ui_vector_size, self.ui_vector_size)
        torch.nn.init.normal_(base_position_embedding, mean=0.0, std=1.0)
        self.base_position_embedding = torch.nn.Parameter(base_position_embedding)
        # -------------------Beat_logical-------------------------
        self.fea2log = nn.Linear(self.ui_vector_size, 2 * self.ui_vector_size)
        self.sigmod = torch.nn.Sigmoid()
        self.projection_regularizer = Regularizer(0.05, 0.05, 1e9)
        self.bn1 = nn.BatchNorm1d(self.ui_vector_size)
        self.Negation = Negation()
        self.center_net = BetaIntersection(self.ui_vector_size)

        # -------------------Beat_logical-------------------------#
    def feature_to_beta(self, feature):
        # logic_input = self.fea2log(self.bn1(feature))
        logic_input = self.fea2log(feature)
        logic_input = self.sigmod(logic_input)
        # logic_input = self.bn2(logic_input)
        logic_input = self.projection_regularizer(logic_input)
        alpha, beta = torch.chunk(logic_input, 2, dim=-1)
        return alpha, beta

    def cal_logit_beta(self, entity_dist, path_dist):
        logit = self.logic_gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, path_dist), p=1, dim=-1)
        return logit

    def _construct_product_logic_embedding(self,embeds):
        item_alpha_input, item_beta_input = self.feature_to_beta(embeds)
        item_logic_output = item_alpha_input / (item_alpha_input + item_beta_input)
        item_alpha_neg, item_beta_neg = self.Negation(item_alpha_input), self.Negation(item_beta_input)
        item_logic_neg_output = item_alpha_neg / (item_alpha_neg + item_beta_neg)
        return item_alpha_input, item_beta_input, item_logic_output, item_alpha_neg, item_beta_neg, item_logic_neg_output

    def calc_logic_KL_loss(self, item_alpha_input,item_beta_input,caps_alpha_input,caps_beta_input,batch_size):

        state_pa, state_pb = self.center_net(caps_alpha_input[:batch_size], caps_beta_input[:batch_size])
        state_dists = torch.distributions.beta.Beta(state_pa, state_pb)
        pos_logic_alpha, pos_logic_beta = item_alpha_input[:batch_size], item_beta_input[:batch_size]
        neg_logic_alpha, neg_logic_beta = item_alpha_input[batch_size:], item_beta_input[batch_size:]
        pos_act_logic_dicts = torch.distributions.beta.Beta(pos_logic_alpha, pos_logic_beta)
        neg_act_logic_dicts = torch.distributions.beta.Beta(neg_logic_alpha, neg_logic_beta)
        positive_logit = self.cal_logit_beta(pos_act_logic_dicts, state_dists)
        negative_logit = self.cal_logit_beta(neg_act_logic_dicts, state_dists)
        bpr = BPRLoss()
        act_kl_loss = bpr(positive_logit, negative_logit)
        return act_kl_loss

    def add_each_calc_logic_KL_loss(self, item_alpha_input,item_beta_input,caps_alpha_input,caps_beta_input,batch_size):
        state_pa, state_pb = self.center_net(caps_alpha_input[:batch_size], caps_beta_input[:batch_size])
        state_dists = torch.distributions.beta.Beta(state_pa, state_pb)
        pos_logic_alpha, pos_logic_beta = item_alpha_input[:batch_size], item_beta_input[:batch_size]
        neg_logic_alpha, neg_logic_beta = item_alpha_input[batch_size:], item_beta_input[batch_size:]
        pos_act_logic_dicts = torch.distributions.beta.Beta(pos_logic_alpha, pos_logic_beta)
        neg_act_logic_dicts = torch.distributions.beta.Beta(neg_logic_alpha, neg_logic_beta)
        # add single interest
        each_state_dists = torch.distributions.beta.Beta(caps_alpha_input[:batch_size,0], caps_beta_input[:batch_size,0])
        each_positive_logit = self.cal_logit_beta(pos_act_logic_dicts, each_state_dists)
        each_positive_logit_set = [each_positive_logit]
        for ind in range(1,self.K):
            each_state_dists = torch.distributions.beta.Beta(caps_alpha_input[:batch_size,ind], caps_beta_input[:batch_size,ind])
            each_positive_logit = self.cal_logit_beta(pos_act_logic_dicts, each_state_dists)
            each_positive_logit_set.append(each_positive_logit)
        # asasasa = torch.stack(each_positive_logit_set,dim=1)
        each_positive_logit_mean = torch.mean(torch.stack(each_positive_logit_set,dim=1),dim=1)
        positive_logit = self.cal_logit_beta(pos_act_logic_dicts, state_dists)
        negative_logit = self.cal_logit_beta(neg_act_logic_dicts, state_dists)
        bpr = BPRLoss()
        act_kl_loss = bpr(positive_logit, negative_logit) + bpr(positive_logit, each_positive_logit_mean) + bpr(each_positive_logit_mean, negative_logit)
        return act_kl_loss
    # -------------------------------------------------------#

    def logic_not(self, vector):
        vector = F.relu(self.not_layer_1(vector))
        vector = self.not_layer_2(vector)
        return vector

    def logic_and(self, vector1, vector2):
        assert(len(vector1.size()) == len(vector2.size()))
        vector = torch.cat((vector1, vector2), dim=(len(vector1.size()) - 1))
        vector = F.relu(self.and_layer_1(vector))
        vector = self.and_layer_2(vector)
        return vector

    def logic_or(self, vector1, vector2):
        assert (len(vector1.size()) == len(vector2.size()))
        vector = torch.cat((vector1, vector2), dim=(len(vector1.size()) - 1))
        vector = F.relu(self.or_layer_1(vector))
        vector = self.or_layer_2(vector)
        return vector

    def purchase_gate(self, uv_vector):
        uv_vector = F.relu(self.purchase_layer_1(uv_vector))
        uv_vector = self.purchase_layer_2(uv_vector)
        return uv_vector

    # def logic_output(self, vector):
    def mse(self, vector1, vector2):
        return ((vector1 - vector2) ** 2).mean()

    def squash(self, caps, bs):
        n = torch.norm(caps, dim=2).view(bs, self.K, 1)
        nSquare = torch.pow(n, 2)

        return (nSquare / ((1 + nSquare) * n + 1e-9)) * caps




    def BetaE_his_B2IRouting(self, his, bs, history_pos_tag):
        """B2I dynamic routing, input behaviors, output caps
        """
        # init b, bji = b[j][i] rather than b[i][j] for matmul convinience
        # no grad for b: https://github.com/Ugenteraan/CapsNet-PyTorch/blob/master/CapsNet-PyTorch.ipynb
        # self.B = torch.nn.init.normal_(torch.empty(self.K, self.L), mean=0.0, std=1.0)
        # B = self.B.detach()
        # B = torch.nn.init.normal_(torch.empty(self.K, his_length), mean=0.0, std=1.0).detach()
        # B = torch.nn.init.normal_(torch.empty(self.K, his_length), mean=0.0, std=1.0).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # except for first routing round, each sample's w is different, so need a dim for batch
        mask = (his != 0).unsqueeze(1).tile(1, self.K, 1)
        drop = (torch.ones_like(mask) * -(1 << 31)).type(torch.float32)
        his = self.iid_embeddings(his.long())  # (bs, L, D)
        his_alpha_input, his_beta_input, his_logic_output, his_alpha_neg, his_beta_neg, his_logic_neg_output = self._construct_product_logic_embedding(his)
        his_alpha_input = history_pos_tag * his_alpha_input + (1 - history_pos_tag) * his_alpha_neg
        his_beta_input = history_pos_tag * his_beta_input + (1 - history_pos_tag) * his_beta_neg
        his_logic_output = history_pos_tag * his_logic_output + (1 - history_pos_tag) * his_logic_neg_output
        # print(his_alpha_input == history_pos_tag * his_alpha_input + (1 - history_pos_tag) * his_alpha_neg)
        # print(his_alpha_input == history_pos_tag * his_alpha_input + (1 - history_pos_tag) * his_alpha_input)
        # caps_alpha_input == history_pos_tag * caps_alpha_input + (1 - history_pos_tag) * caps_alpha_neg
        # caps_alpha_input == history_pos_tag * caps_alpha_input + (1 - history_pos_tag) * caps_alpha_neg

        B = torch.tile(self.base_B, (bs, 1, 1))  # (bs, K, D)
        B = torch.matmul(B, torch.transpose(his_logic_output, 1, 2))# (bs, K, L)
        # masking, make padding indices' routing logit as INT_MAX so that softmax result is 0
        # (bs, L) -> (bs, 1, L) -> (bs, K, L)


        # his = torch.matmul(his, self.S)
        his_alpha = torch.matmul(his_alpha_input, self.S)
        his_alpha = self.projection_regularizer(self.sigmod(his_alpha))
        # his_alpha = self.sigmod(his_alpha)
        his_beta = torch.matmul(his_beta_input, self.S)
        his_beta = self.projection_regularizer(self.sigmod(his_beta))
        # his_beta = self.sigmod(his_beta)
        his_fuse_logic = his_alpha / (his_alpha + his_beta)
        for i in range(self.R):
            BMasked = torch.where(mask, B, drop)
            W = self.sigmod(torch.softmax(BMasked, dim=2)) # (bs, K, L)
            if i < self.R - 1:
                # with torch.no_grad():
                    # weighted sum all i to each j
                caps_alpha = torch.matmul(W, his_alpha) # (bs, K, D)
                caps_alpha = self.projection_regularizer(self.sigmod(caps_alpha))
                # caps_alpha = self.sigmod(caps_alpha)
                # caps_alpha = self.squash(caps_alpha, bs)
                caps_beta = torch.matmul(W, his_beta) # (bs, K, D)
                caps_beta = self.projection_regularizer(self.sigmod(caps_beta))
                # caps_beta = self.sigmod(caps_beta)
                # caps_beta = self.squash(caps_beta, bs)
                caps = caps_alpha / (caps_alpha + caps_beta)
                caps = self.squash(caps, bs)
                # caps = torch.matmul(W, his) # (bs, K, D)
                # caps = self.squash(caps, bs)
                B += torch.matmul(caps, torch.transpose(his_fuse_logic, 1, 2)) # (bs, K, L)
            else:
                caps_alpha = torch.matmul(W, his_alpha) # (bs, K, D)
                caps_alpha = self.projection_regularizer(self.sigmod(caps_alpha))
                # caps_alpha = self.sigmod(caps_alpha)
                caps_beta = torch.matmul(W, his_beta) # (bs, K, D)
                caps_beta = self.projection_regularizer(self.sigmod(caps_beta))
                # caps_beta = self.sigmod(caps_beta)
                caps = caps_alpha / (caps_alpha + caps_beta)
                caps = self.squash(caps, bs)
                # skip routing logits update in last round
        # mlp
        # caps = self.dense2(torch.relu(self.dense1(caps)))  # (bs, K, D)
        # caps_alpha_input, caps_beta_input, caps_logic_output, caps_alpha_neg, caps_beta_neg, caps_logic_neg_output = self._construct_product_logic_embedding(caps)
        # print(caps_alpha_input == history_pos_tag * caps_alpha_input + (1 - history_pos_tag) * caps_alpha_neg)
        # print(caps_alpha_input == history_pos_tag * caps_alpha_input + (1 - history_pos_tag) * caps_alpha_input)
        ## l2 norm
        # caps = caps / (th.norm(caps, dim=2).view(bs, self.K, 1) + 1e-9)

        return caps_alpha, caps_beta, caps

    def BetaE_MI_Self_Attentive(self, his, add_pos, history_pos_tag):
        mask = (his != 0).unsqueeze(1).tile(1, self.K, 1)
        drop = (torch.ones_like(mask) * -(1 << 31)).type(torch.float32)
        item_list_emb = self.iid_embeddings(his.long()) # (bs, L, D)
        his_alpha_input, his_beta_input, his_logic_output, his_alpha_neg, his_beta_neg, his_logic_neg_output = self._construct_product_logic_embedding(item_list_emb)
        his_alpha_input = history_pos_tag * his_alpha_input + (1 - history_pos_tag) * his_alpha_neg
        his_beta_input = history_pos_tag * his_beta_input + (1 - history_pos_tag) * his_beta_neg
        his_logic_output = history_pos_tag * his_logic_output + (1 - history_pos_tag) * his_logic_neg_output
        position_embedding = torch.tile(self.base_position_embedding, (item_list_emb.size(0), 1, 1))
        position_emb = torch.matmul(his_logic_output,position_embedding)

        if add_pos:
            # position_emb = position_embedding.expand(item_list_emb.size(0), -1, -1)
            # item_list_add_pos = his_logic_output + position_emb
            item_list_add_pos_alpha = his_alpha_input + position_emb
            item_list_add_pos_beta = his_beta_input + position_emb

        else:
            # item_list_add_pos = his_logic_output
            item_list_add_pos_alpha = his_alpha_input
            item_list_add_pos_beta = his_beta_input

        item_hidden_alpha = torch.tanh(self.dense1(item_list_add_pos_alpha))
        item_att_w_alpha = self.dense3(item_hidden_alpha).permute(0, 2, 1)
        item_att_w_alpha = torch.where(mask, item_att_w_alpha, drop)
        item_att_w_alpha = F.softmax(item_att_w_alpha, dim=-1)
        interest_emb_alpha = torch.matmul(item_att_w_alpha, his_logic_output)
        interest_emb_alpha = self.projection_regularizer(self.sigmod(interest_emb_alpha))

        item_hidden_beta = torch.tanh(self.dense1(item_list_add_pos_beta))
        item_att_w_beta = self.dense3(item_hidden_beta).permute(0, 2, 1)
        item_att_w_beta = torch.where(mask, item_att_w_beta, drop)
        item_att_w_beta = F.softmax(item_att_w_beta, dim=-1)
        interest_emb_beta = torch.matmul(item_att_w_beta, his_logic_output)
        interest_emb_beta = self.projection_regularizer(self.sigmod(interest_emb_beta))

        interest_emb = interest_emb_alpha/(interest_emb_alpha + interest_emb_beta)

        # self.user_eb = interest_emb
        #
        # atten = torch.matmul(self.user_eb, self.item_eb.view(-1, self.dim, 1)).view(item_list_emb.size(0),
        #                                                                             self.num_interest)
        # atten = F.softmax(atten.pow(1), dim=-1)
        #
        # readout_indices = torch.argmax(atten, dim=1) + torch.arange(item_list_emb.size(0)) * self.num_interest
        # readout = self.user_eb.view(-1, self.dim)[readout_indices]

        return interest_emb_alpha, interest_emb_beta, interest_emb


    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0].long()
        i_ids = feed_dict['X'][:, 1].long()

        feed_dict[global_p.C_HISTORY] = feed_dict[global_p.C_HISTORY][:, ~torch.any(feed_dict[global_p.C_HISTORY] == -1, dim=0)]
        feed_dict[global_p.C_HISTORY_POS_TAG] = feed_dict[global_p.C_HISTORY_POS_TAG][:, ~torch.any(feed_dict[global_p.C_HISTORY_POS_TAG] == -1, dim=0)]
        history = feed_dict[global_p.C_HISTORY]
        # print(history.size())
        batch_size, his_length = list(history.size())


        history_pos_tag = feed_dict[global_p.C_HISTORY_POS_TAG].unsqueeze(2).float()

        # user/item vectors shape: (batch_size, embedding_size)
        # user_vectors = self.uid_embeddings(u_ids)
        item_vectors = self.iid_embeddings(i_ids)

        item_alpha_input, item_beta_input, item_vectors, item_alpha_neg, item_beta_neg, item_logic_neg_output = self._construct_product_logic_embedding(item_vectors)


        # history item purchase hidden factors shape: (batch, user, embedding)
        # his_vectors = self.iid_embeddings(history.long())

        # ____________________MI_capsule (bs, K, D)_____________
        # caps = self.B2IRouting(history.long(), batch_size,his_length)
        # caps_alpha_input, caps_beta_input, caps = self.BetaE_his_B2IRouting(history.long(), batch_size,history_pos_tag)# B2IRouting
        caps_alpha_input, caps_beta_input, caps = self.BetaE_MI_Self_Attentive(history.long(), 1, history_pos_tag)# Self_Attentive
        bsize = int(feed_dict['Y'].shape[0] / 2)
        # caps = self.MI_Self_Attentive(history.long(), 1)
        constraint = list([caps])
        # --------------------multi-interest > single interest--------------------------------------
        each_caps = self.logic_not(caps)
        each_sent_vector = [self.logic_or(each_caps[:,0], item_vectors)]
        for ind in range(1,self.K):
            each_sent_vector.append(self.logic_or(each_caps[:,ind], item_vectors))
        each_sent_vector = torch.stack(each_sent_vector,dim=1)
        each_prediction = torch.mean(F.cosine_similarity(each_sent_vector, self.true.view([1, -1]), dim=-1),dim=1) * 10
        # -----------------------------------------------------------------------------------------
        tmp_vector = self.logic_not(caps[:, 0])
        shuffled_caps_idx = [i for i in range(1, self.K)]
        np.random.shuffle(shuffled_caps_idx)
        for i in shuffled_caps_idx:
            tmp_vector = self.logic_or(tmp_vector, self.logic_not(caps[:, i]))
            constraint.append(tmp_vector.view(batch_size, -1, self.ui_vector_size))
        left_vector = tmp_vector
        right_vector = item_vectors
        constraint.append(right_vector.view(batch_size, -1, self.ui_vector_size))
        sent_vector = self.logic_or(left_vector, right_vector)
        constraint.append(sent_vector.view(batch_size, -1, self.ui_vector_size))
        # check_list.append(('sent_vector', sent_vector))
        if feed_dict['rank'] == 1:
            prediction = F.cosine_similarity(sent_vector, self.true.view([1, -1])) * 10
        else:
            prediction = F.cosine_similarity(sent_vector, self.true.view([1, -1])) * \
                         (self.label_max - self.label_min) / 2 + (self.label_max + self.label_min) / 2
        constraint = torch.cat(constraint, dim=1)
        out_dict = {'prediction': prediction,
                    'each_prediction': each_prediction,
                    # 'predict_purchase': predict_purchase,
                    # 'his_vectors': his_vectors,
                    'check': check_list,
                    'constraint': constraint,
                    'interim': left_vector,
                    'caps_alpha': caps_alpha_input,
                    'caps_beta': caps_beta_input,
                    'caps': caps,
                    'item_alpha': item_alpha_input,
                    'item_beta': item_beta_input,
                    'item': item_vectors
                    }
        return out_dict


    def forward(self, feed_dict):
        """
        除了预测之外，还计算loss
        :param feed_dict: 型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果，loss是损失
        """
        out_dict = self.predict(feed_dict)
        batch_size = int(feed_dict['Y'].shape[0] / 2)
        caps_alpha_input, caps_beta_input, caps = out_dict['caps_alpha'], out_dict['caps_beta'], out_dict['caps']
        item_alpha_input, item_beta_input, item_vectors = out_dict['item_alpha'], out_dict['item_beta'], out_dict['item']
        # KL_loss = self.calc_logic_KL_loss(item_alpha_input, item_beta_input, caps_alpha_input, caps_beta_input, batch_size) * self.logic_weight
        KL_loss = self.add_each_calc_logic_KL_loss(item_alpha_input, item_beta_input, caps_alpha_input, caps_beta_input,batch_size) * self.logic_weight
        check_list = out_dict['check']
        # predict_purchase = out_dict['predict_purchase']
        false = self.logic_not(self.true).view(1, -1)
        # his_vectors = out_dict['his_vectors']
        constraint = out_dict['constraint']

        # regularizer
        dim = len(constraint.size())-1

        # length constraint
        # r_length = constraint.norm(dim=dim)()

        # not
        # r_not_true = self.mse(self.logic_not(self.logic_not(self.true)), self.true) / (self.true ** 2).mean()
        # r_not_true = (1 - (F.cosine_similarity(self.logic_not(self.logic_not(self.true)), self.true, dim=0)
        #               * self.sim_scale).sigmoid()).sum()
        r_not_not_true = (1 - F.cosine_similarity(self.logic_not(self.logic_not(self.true)), self.true, dim=0)).sum()
        # check_list.append(('r_not_not_true', r_not_not_true))
        # r_not_self = self.mse(self.logic_not(self.logic_not(constraint)), constraint) / (constraint ** 2).mean()
        # r_not_self = (F.cosine_similarity(
        #     self.logic_not(self.logic_not(constraint)), constraint, dim=dim)
        #               * self.sim_scale).sigmoid().mean()
        r_not_not_self = \
            (1 - F.cosine_similarity(self.logic_not(self.logic_not(constraint)), constraint, dim=dim)).mean()
        # check_list.append(('r_not_not_self', r_not_not_self))
        r_not_self = (1 + F.cosine_similarity(self.logic_not(constraint), constraint, dim=dim)).mean()

        r_not_self = (1 + F.cosine_similarity(self.logic_not(constraint), constraint, dim=dim)).mean()

        r_not_not_not = \
            (1 + F.cosine_similarity(self.logic_not(self.logic_not(constraint)), self.logic_not(constraint), dim=dim)).mean()

        # and
        # r_and_true = self.mse(
        #     self.logic_and(constraint, self.true.expand_as(constraint)), constraint) / (constraint ** 2).mean()
        # r_and_true = (-F.cosine_similarity(
        #     self.logic_and(constraint, self.true.expand_as(constraint)), constraint, dim=dim)
        #               * self.sim_scale).sigmoid().mean()
        r_and_true = (1 - F.cosine_similarity(
            self.logic_and(constraint, self.true.expand_as(constraint)), constraint, dim=dim)).mean()
        # check_list.append(('r_and_true', r_and_true))
        # r_and_false = self.mse(self.logic_and(constraint, false.expand_as(constraint)), false) / (false ** 2).mean()
        # r_and_false = (-F.cosine_similarity(
        #     self.logic_and(constraint, false.expand_as(constraint)), false.expand_as(constraint), dim=dim)
        #                * self.sim_scale).sigmoid().mean()
        r_and_false = (1 - F.cosine_similarity(
            self.logic_and(constraint, false.expand_as(constraint)), false.expand_as(constraint), dim=dim)).mean()
        # check_list.append(('r_and_false', r_and_false))
        # r_and_self = self.mse(self.logic_and(constraint, constraint), constraint) / (constraint ** 2).mean()
        r_and_self = (1 - F.cosine_similarity(self.logic_and(constraint, constraint), constraint, dim=dim)).mean()
        # check_list.append(('r_and_self', r_and_self))

        # NEW ADDED REG NEED TO TEST
        r_and_not_self = (1 - F.cosine_similarity(
            self.logic_and(constraint, self.logic_not(constraint)), false.expand_as(constraint), dim=dim)).mean()
        # check_list.append(('r_and_not_self', r_and_not_self))
        r_and_not_self_inverse = (1 - F.cosine_similarity(
            self.logic_and(self.logic_not(constraint), constraint), false.expand_as(constraint), dim=dim)).mean()
        # check_list.append(('r_and_not_self_inverse', r_and_not_self_inverse))

        # or
        # r_or_true = self.mse(
        #     self.logic_or(constraint, self.true.expand_as(constraint)), self.true) / (self.true ** 2).mean()
        # r_or_true = (-F.cosine_similarity(
        #     self.logic_or(constraint, self.true.expand_as(constraint)), self.true.expand_as(constraint), dim=dim)
        #              * self.sim_scale).sigmoid().mean()
        r_or_true = (1 - F.cosine_similarity(
            self.logic_or(constraint, self.true.expand_as(constraint)), self.true.expand_as(constraint), dim=dim))\
            .mean()
        # check_list.append(('r_or_true', r_or_true))
        # r_or_false = self.mse(
        #     self.logic_or(constraint, false.expand_as(constraint)), constraint) / (constraint ** 2).mean()
        # r_or_false = (-F.cosine_similarity(self.logic_or(constraint, false.expand_as(constraint)), constraint, dim=dim)
        #               * self.sim_scale).sigmoid().mean()
        r_or_false = (1 - F.cosine_similarity(
            self.logic_or(constraint, false.expand_as(constraint)), constraint, dim=dim)).mean()
        # check_list.append(('r_or_false', r_or_false))
        # r_or_self = self.mse(self.logic_or(constraint, constraint), constraint) / (constraint ** 2).mean()
        # r_or_self = (-F.cosine_similarity(self.logic_or(constraint, constraint), constraint, dim=dim)
        #              * self.sim_scale).sigmoid().mean()
        r_or_self = (1 - F.cosine_similarity(self.logic_or(constraint, constraint), constraint, dim=dim)).mean()
        # check_list.append(('r_or_self', r_or_self))

        r_or_not_self = (1 - F.cosine_similarity(
            self.logic_or(constraint, self.logic_not(constraint)), self.true.expand_as(constraint), dim=dim)).mean()
        r_or_not_self_inverse = (1 - F.cosine_similarity(
            self.logic_or(self.logic_not(constraint), constraint), self.true.expand_as(constraint), dim=dim)).mean()
        # check_list.append(('r_or_not_self', r_or_not_self))
        # check_list.append(('r_or_not_self_inverse', r_or_not_self_inverse))

        # True/False
        true_false = 1 + F.cosine_similarity(self.true, false.view(-1), dim=0)

        r_loss = r_not_not_true + r_not_not_self + r_not_self + \
                 r_and_true + r_and_false + r_and_self + r_and_not_self + r_and_not_self_inverse + \
                 r_or_true + r_or_false + r_or_self + true_false + r_or_not_self + r_or_not_self_inverse + r_not_not_not
        r_loss = r_loss * self.r_weight

        # pos_loss = None
        # recommendation loss
        if feed_dict['rank'] == 1:
            batch_size = int(feed_dict['Y'].shape[0] / 2)
            # tf_matrix = self.true.view(1, -1).expand(batch_size, -1)
            pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
            # pos_loss = 10 - torch.mean(pos)
            loss = -(pos - neg).sigmoid().log().sum()
            each_loss = -(out_dict['prediction'][:batch_size] - out_dict['each_prediction'][:batch_size]).sigmoid().log().sum() * self.logic_weight #multi-interest > single interest
            # check_list.append(('bpr_loss', loss))
        else:
            loss = torch.nn.MSELoss()(out_dict['prediction'], feed_dict['Y'])

        # predict_purchase_loss = (2 - predict_purchase)
        loss = loss + r_loss + KL_loss + each_loss #+ self.ppl_weight * predict_purchase_loss #+ self.pos_weight * pos_loss
        # check_list.append(('r_loss', r_loss))
        out_dict['loss'] = loss
        out_dict['check'] = check_list
        return out_dict


