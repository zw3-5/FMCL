import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender

from recbole.model.sequential_recommender.bsarec import BSARecEncoder

from recbole.model.loss import BPRLoss


class FMCLRec(SequentialRecommender):
    def __init__(self, config, dataset,args):
        super(FMCLRec, self).__init__(config, dataset)
        
        self.args = args
        # load parameters info
        self.num_hidden_layers = args.num_hidden_layers
        self.num_attention_heads = args.num_attention_heads

        self.hidden_size = args.hidden_size  # same as embedding_size
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.attn_dropout_prob = args.attention_probs_dropout_prob
        self.hidden_act = args.hidden_act
        self.initializer_range = args.initializer_range
        
        self.batch_size = args.train_batch_size
        self.lmd = args.lmd
        self.beta=args.beta
        self.tau = args.tau
        self.sim = args.sim
        self.use_rl=args.use_rl

        self.layer_norm_eps = config['layer_norm_eps']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items +1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        self.trm_encoder = BSARecEncoder(self.args)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def contrast(self, z_i, z_j, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size=z_i.shape[0]
        temp=self.tau
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)
        return positive_samples,negative_samples

    def meta_contrast_rl(self,a, b):
        ori_p, ori_n = self.contrast(a, b, "dot")
        min_positive_value, min_pos_pos = torch.min(ori_p, dim=-1)
        max_negative_value, max_neg_pos = torch.max(ori_n, dim=-1)
        lgamma_margin_pos, _ = torch.min(torch.cat((min_positive_value.unsqueeze(1), max_negative_value
                                                    .unsqueeze(1)), dim=1), dim=-1)
        lgamma_margin_pos = lgamma_margin_pos.unsqueeze(1)
        lgamma_margin_neg, _ = torch.max(torch.cat((min_positive_value.unsqueeze(1), max_negative_value
                                                    .unsqueeze(1)), dim=1), dim=-1)
        lgamma_margin_neg = lgamma_margin_neg.unsqueeze(1)
        loss = torch.mean(torch.clamp(ori_p - lgamma_margin_pos, min=0))
        loss += torch.mean(torch.clamp(lgamma_margin_neg - ori_n, min=0))
        return loss

    def meta_rl(self,sequence_output_0, sequence_output_1,sequence_output_2,sequence_output_3):
        rl_loss=0.
        rl_loss+=self.meta_contrast_rl(sequence_output_0,sequence_output_3)
        rl_loss+=self.meta_contrast_rl(sequence_output_1,sequence_output_2)
        rl_loss+=self.meta_contrast_rl(sequence_output_2,sequence_output_3)
        return 0.1*rl_loss

    def meta_contrast(self, sequence_output_0, sequence_output_1,meta_aug, mode):
        aug_1, aug_2 = meta_aug
        use_rl = self.use_rl
    
        # 生成增强视图
        sequence_output_2 = aug_1(sequence_output_0)
        sequence_output_3 = aug_2(sequence_output_1)
    
        # 计算 DCL 对比损失
        cl_loss_0 = self.dcl_loss(sequence_output_0, sequence_output_3, temperature=self.tau, lambda_neg=1.0)
        cl_loss_1 = self.dcl_loss(sequence_output_1, sequence_output_2, temperature=self.tau, lambda_neg=1.0)
        cl_loss_2 = self.dcl_loss(sequence_output_2, sequence_output_3, temperature=self.tau, lambda_neg=1.0)
    
        cl_loss = cl_loss_0 + cl_loss_1 + cl_loss_2
    
        if use_rl:
            cl_loss += self.meta_rl(sequence_output_0, sequence_output_1, sequence_output_2, sequence_output_3)
    
        return cl_loss


    def calculate_loss(self, interaction,meta_aug=None,mode=None):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:
            if mode=="step2":
                loss=0.
            else:
                test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = self.loss_fct(logits, pos_items)

        aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = \
            interaction['aug1'], interaction['aug_len1'], interaction['aug2'], interaction['aug_len2']
        seq_output1 = self.forward(aug_item_seq1, aug_len1)
        seq_output2 = self.forward(aug_item_seq2, aug_len2)
        nce_loss = self.dcl_loss(seq_output1, seq_output2, temperature=self.tau, lambda_neg=1.0)


        loss += self.lmd * nce_loss
        if mode == "step1":
            loss += self.beta * self.meta_contrast(seq_output1, seq_output2, meta_aug, mode)
            return loss
        elif mode == "step2":
            loss += self.meta_contrast(seq_output1, seq_output2, meta_aug, mode)
            return loss
        else:
            loss += self.beta * self.meta_contrast(seq_output1, seq_output2, meta_aug, mode)
            return loss

    def dcl_loss(self, z1, z2, temperature=0.1, lambda_neg=1.0):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        batch_size = z1.shape[0]
        representations = torch.cat([z1, z2], dim=0)
    
        sim_pos = F.cosine_similarity(z1, z2, dim=-1)
        loss_pos = 1 - sim_pos
    
        negatives = representations.detach()
        sim_neg_z1 = torch.matmul(z1, negatives.T) / temperature
        sim_neg_z2 = torch.matmul(z2, negatives.T) / temperature
    
        mask = ~torch.eye(2 * batch_size, device=z1.device).bool()
        sim_neg_z1 = sim_neg_z1.masked_select(mask[:batch_size]).reshape(batch_size, -1)
        sim_neg_z2 = sim_neg_z2.masked_select(mask[batch_size:]).reshape(batch_size, -1)
    
        loss_neg = (sim_neg_z1.mean(dim=-1) + sim_neg_z2.mean(dim=-1)) / 2
        loss = loss_pos + lambda_neg * loss_neg
        return loss.mean()

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
        
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
