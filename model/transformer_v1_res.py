# This file is part of Lingo3DMol
#
# Lingo3DMol is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Lingo3DMol is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Lingo3DMol. If not, see <https://www.gnu.org/licenses/>.



import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math
import numpy as np
import time
from copy import deepcopy
from model.Module import MLP,MultiHeadedAttention,MultiHeadedAttention_att,MultiHeadedAttentionBias\
    ,PositionwiseFeedForward,PositionalEncoding,EncoderLayer,Encoder,DecoderLayerBias,DecoderBias,edge_vector,cdist

from model.transformer_v1_res_mp1 import subsequent_mask,topkp_random,find_root_smi_cur,segment_coords,next_coords

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size=12, gt_vocab_size=76, dropout=0.1):
        super(TransformerModel, self).__init__()

        # Basic parameters
        self.src_vocab_size = 12
        self.tgt_size = 100
        self.coords_size = 99
        self.h = 8
        self.grid_size = 240
        d_model = 512
        self.d_model=d_model
        
        # Embeddings
        self.coords_emb = nn.Embedding(self.grid_size, 128)
        self.coords_emb_gt = nn.Embedding(self.grid_size, 128)
        self.residue = nn.Embedding(50, 128 * 3)
        self.anchor_emb = nn.Embedding(20, 128)
        self.relative_emb = edge_vector()
        self.dist_emb = cdist()
        self.relative_emb_topo = edge_vector()
        self.dist_emb_topo = cdist()

        self.src_fea = nn.Embedding(src_vocab_size, 128)
        self.gt_fea = nn.Embedding(gt_vocab_size, 128)
        self.gt_fea1 = nn.Embedding(gt_vocab_size, 128)

        # Edge linear layers
        self.linears_edge = MLP(in_features=64, hidden_layer_sizes=[64, 32], out_features=self.h, dropout_p=0.1)
        self.linears_edge1 = nn.Linear(self.h, self.h)
        self.linears_edge_topo = MLP(in_features=64, hidden_layer_sizes=[64, 32], out_features=self.h, dropout_p=0.1)
        self.linears_edge1_topo = nn.Linear(self.h, self.h)

        # Attention and Feed Forward layers
        c = copy.deepcopy
        
        attn = MultiHeadedAttention(h=8, d_model=d_model)
        attn_att = MultiHeadedAttention_att(h=8, d_model=d_model)
        attnbias = MultiHeadedAttentionBias(h=8, d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=2048, dropout=dropout)

        # Positional encoding and encoder
        self.pos_encode = PositionalEncoding(d_model, dropout)
        self.encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        encoder_layer_num = 3
        self.encoder = Encoder(self.encoder_layer, encoder_layer_num)

        # Type decoder
        self.decoder_layer = DecoderLayerBias(d_model, c(attnbias), c(attn_att), c(ff), dropout)
        decoder_layer_num = 3
        self.decoder = DecoderBias(self.decoder_layer, decoder_layer_num)

        # Relative coords decoder
        self.decoder_layerbias = DecoderLayerBias(d_model, c(attnbias), c(attn_att), c(ff), dropout)
        decoder_layer_num = 3
        self.decoder_relative = DecoderBias(self.decoder_layerbias, decoder_layer_num)

        # GPT atom features
        self.proj = nn.Linear(d_model, gt_vocab_size)
        self.proj_aux = nn.Linear(d_model, gt_vocab_size)

        self.proj2_aux = MLP(in_features=2 * d_model + 128, hidden_layer_sizes=[256] * 2, out_features=13, dropout_p=0.1)
        self.proj3_aux = MLP(in_features=3 * d_model + 128, hidden_layer_sizes=[256] * 2, out_features=181, dropout_p=0.1)
        self.proj4_aux = MLP(in_features=4 * d_model + 128, hidden_layer_sizes=[256] * 2, out_features=181, dropout_p=0.1)

        self.proj2 = MLP(in_features=2 * d_model + 128, hidden_layer_sizes=[256] * 2, out_features=13, dropout_p=0.1)
        self.proj3 = MLP(in_features=3 * d_model + 128, hidden_layer_sizes=[256] * 2, out_features=181, dropout_p=0.1)
        self.proj4 = MLP(in_features=4 * d_model + 128, hidden_layer_sizes=[256] * 2, out_features=181, dropout_p=0.1)

        # Coordinate MLPs
        self.fcx = MLP(in_features=d_model, hidden_layer_sizes=[256] * 2, out_features=self.grid_size, dropout_p=0.1)
        self.fcy = MLP(in_features=d_model + self.grid_size, hidden_layer_sizes=[256] * 2, out_features=self.grid_size, dropout_p=0.1)
        self.fcz = MLP(in_features=d_model + 2 * self.grid_size, hidden_layer_sizes=[256] * 2, out_features=self.grid_size, dropout_p=0.1)

        # Matrix and contact        # projection layers
        self.proj_matrix = MLP(in_features=8, hidden_layer_sizes=[32] * 2, out_features=11, dropout_p=0.1)
        self.proj_matrix1 = MLP(in_features=8, hidden_layer_sizes=[32] * 2, out_features=11, dropout_p=0.1)
        self.proj_contact = nn.Sequential(nn.Linear(1024, d_model), nn.Linear(d_model, 1), nn.Sigmoid())
        self.proj_residue = nn.Linear(1024, d_model)

    def prep_enc_input(self, coords, atom_type, residue, critical_anchor, src_mask):
        coords_emb = self.coords_emb(coords)
        coords_embedding = torch.reshape(coords_emb, (coords_emb.shape[0], coords_emb.shape[1], -1))
        
        type_emb = self.src_fea(atom_type)
        total_feature = torch.cat((coords_embedding, type_emb), dim=-1)
        
        residue_emb = self.residue(residue)
        critical_anchor_emb = self.anchor_emb(critical_anchor)
        residue_total_emb = torch.cat([residue_emb, critical_anchor_emb], dim=-1)
        
        total_feature1 = total_feature + residue_total_emb
        final_fea = self.pos_encode(total_feature1)
        
        src_mask1 = src_mask.repeat(1, src_mask.size(-1), 1)
        
        return final_fea, src_mask1


    def prepare_topo_tensors(self, captions, gt_coords):
        new_fea0_   = self.gt_fea1(captions)
        gt_coords_emb0 = self.coords_emb_gt(gt_coords)
        gt_coords_emb  = gt_coords_emb0.reshape(gt_coords_emb0.shape[0], gt_coords_emb0.shape[1], -1)
        new_fea0 = torch.cat([new_fea0_, gt_coords_emb], dim=-1)
        tgt_fea  = self.pos_encode(new_fea0)
        tgt_mask = subsequent_mask(self.tgt_size).long().cuda()
        edge_vec_topo = self.relative_emb_topo(gt_coords.float() / 10.0, gt_coords.float() / 10.0)
        dist_sca_topo = self.dist_emb_topo(gt_coords.float() / 10.0, gt_coords.float() / 10.0)
        edge_fea_topo = torch.cat([edge_vec_topo, dist_sca_topo], dim=-1)
        edge_fea_bias_topo = self.linears_edge1_topo(self.linears_edge_topo(edge_fea_topo)).permute(0, 3, 1, 2)
        return new_fea0_,new_fea0, tgt_fea, tgt_mask, edge_fea_bias_topo

    def forward(self, coords, atom_type, residue,critical_anchor,src_mask, captions=None, gt_coords=None, smi_map=None, smi_map_n1=None, smi_map_n2=None,isTrain=True, contact=None, sample_num=1, max_step=100, isMultiSample=False):
        '''
        :param coords: protein coordinates [Batch, SeqNum, 3]
        :param atom_type: protein atom type [Batch, SeqNum]
        :param residue: hba_hbd+aromatic [Batch, SeqNum]
        :param critical_anchor: 0,1 whether it is a critical anchor point (nci+4A) [Batch, SeqNum]
        :param src_mask: SeqNum mask 0,1
        :param captions: ligand sequence [Batch, SeqNum]
        :param gt_coords: ligand coordinates [Batch, SeqNum, 3]
        :param smi_map: root indices [Batch, SeqNum]
        :param smi_map_n1: root root indices [Batch, SeqNum]
        :param smi_map_n2: root root root indices [Batch, SeqNum]
        :param isTrain: true for training, false for evaluation
        :param contact: for inference, nci only takes the first anchor point
        :param sample_num: number of samples for diversity sampling
        :param max_step: default ligand length 100
        :return:
        '''
        src_mask  = src_mask.unsqueeze(1)

        final_fea, src_mask_enc = self.prep_enc_input(coords,atom_type,residue,critical_anchor,src_mask)
        memory    = self.encoder(final_fea, src_mask_enc)

        if isTrain:
            #topo: non bias att
            token_mask1 = torch.where(captions == 0, 0, 1).unsqueeze(-1).repeat(1, 1, atom_type.size(1))
            src_mask2   = src_mask.repeat(1, captions.size(1), 1) * token_mask1
            
            new_fea0_,new_fea0, tgt_fea, tgt_mask, edge_fea_bias_topo = self.prepare_topo_tensors(captions, gt_coords)

            res,att            = self.decoder(tgt_fea, memory, src_mask2, tgt_mask,edge_fea_bias_topo)

            #cross attention map constrain
            att1        = att.permute(0,2,3,1)
            cross_prob  = F.log_softmax(self.proj_matrix(att1),dim=-1)[:,1:,:]
            cross_label = torch.clamp(torch.sqrt(torch.sum(torch.square(gt_coords[:,1:,None,:]/10.0 - coords[:,None,...]/10.0),dim=-1)),min=0,max=10).long()

            # first result: 分子拓扑
            proj = self.proj(res)

            #second part: predict local coords 
            target_type = new_fea0_[:, 1:]

            #indices
            batch_ind_total = torch.unsqueeze(torch.range(0, smi_map.shape[0] - 1).cuda(), dim=-1).repeat(
                (1, smi_map[:, 1:].shape[1]))
            batch_ind = batch_ind_total.long()

            root_idx = smi_map[:, 1:]
            root_root_idx = smi_map_n1[:, 1:]
            root_root_root_idx = smi_map_n2[:, 1:]

            #features
            new_res_aux = res[:, :-1]

            root_fea_aux = new_res_aux[batch_ind, root_idx]
            root_root_fea_aux = new_res_aux[batch_ind, root_root_idx]
            root_root_root_fea_aux = new_res_aux[batch_ind, root_root_root_idx]

            feature_dist_aux = torch.cat((target_type, new_res_aux, root_fea_aux), dim=-1)
            dist_pred_aux = F.log_softmax(self.proj2_aux(feature_dist_aux), dim=-1)
            feature_theta_aux = torch.cat((target_type, new_res_aux, root_fea_aux, root_root_fea_aux), dim=-1)
            theta_pred_aux = F.log_softmax(self.proj3_aux(feature_theta_aux), dim=-1)
            feature_degree_aux = torch.cat((target_type, new_res_aux, root_fea_aux, root_root_fea_aux, root_root_root_fea_aux), dim=-1)
            degree_pred_aux = F.log_softmax(self.proj4_aux(feature_degree_aux), dim=-1)


            '''
            relative coords
            '''

            root_coords = gt_coords[:, :-1][batch_ind, root_idx]
            gt_coords_emb_rooted = self.coords_emb_gt(root_coords)
            gt_coords_emb_rooted1 = gt_coords_emb_rooted.reshape(gt_coords_emb_rooted.shape[0],gt_coords_emb_rooted.shape[1], -1)
            new_fea2 = torch.cat([target_type, gt_coords_emb_rooted1], dim=-1)
            tgt_fea2 = self.pos_encode(new_fea2)
            src_mask3 = src_mask.repeat(1, captions.size(1) - 1, 1)
            
            tgt_mask2 = subsequent_mask(self.tgt_size - 1).long().cuda()



            edge_vec = self.relative_emb(root_coords.float() / 10.0, root_coords.float() / 10.0)
            dist_sca = self.dist_emb(root_coords.float() / 10.0, root_coords.float() / 10.0)
            edge_fea = torch.cat([edge_vec, dist_sca], dim=-1)
            edge_fea_bias = self.linears_edge1(self.linears_edge(edge_fea)).permute(0, 3, 1, 2)
            res_aux,att_relative= self.decoder_relative(tgt_fea2, memory, src_mask3, tgt_mask2,edge_fea_bias,None)
            att1_relative = att_relative.permute(0, 2, 3, 1)
            cross_prob_relative = F.log_softmax(self.proj_matrix1(att1_relative), dim=-1)
            cross_label_relative = torch.clamp(torch.sqrt(torch.sum(torch.square(gt_coords[:, 1:, None, :] / 10.0 - coords[:, None, ...] / 10.0), dim=-1)), min=0,max=10).long()

            new_res = res_aux
            root_fea = new_res[batch_ind, root_idx]
            root_root_fea = new_res[batch_ind, root_root_idx]
            root_root_root_fea = new_res[batch_ind, root_root_root_idx]

            feature_dist = torch.cat((target_type, new_res, root_fea), dim=-1)
            dist_pred = F.log_softmax(self.proj2(feature_dist), dim=-1)
            feature_theta = torch.cat((target_type, new_res, root_fea, root_root_fea), dim=-1)
            theta_pred = F.log_softmax(self.proj3(feature_theta), dim=-1)
            feature_degree = torch.cat((target_type, new_res, root_fea, root_root_fea, root_root_root_fea), dim=-1)
            degree_pred = F.log_softmax(self.proj4(feature_degree), dim=-1)

            '''
            5) third att with gt ... get final acc coords
            '''
            pos_res    = new_res+tgt_fea2
            pos_res1   = pos_res
            x_p_hidden = self.fcx(pos_res1)
            x_p        = F.log_softmax(x_p_hidden,dim=-1)

            x_p_feature = torch.cat([pos_res1,x_p_hidden],dim=-1)
            y_p_hidden  =self.fcy(x_p_feature)
            y_p         = F.log_softmax(y_p_hidden,dim=-1)

            y_p_feature = torch.cat([x_p_feature,y_p_hidden],dim=-1)
            z_p = F.log_softmax(self.fcz(y_p_feature), dim=-1)
            p_coords = torch.stack((x_p, y_p, z_p), dim=-2)

            return proj,p_coords,dist_pred,dist_pred_aux,theta_pred,theta_pred_aux,degree_pred,degree_pred_aux

        else:

            with torch.no_grad():

                #turn max step to tensor
                max_step = torch.tensor(max_step).cuda()
                residue      = residue.repeat(sample_num, 1)
                residue_mask    = torch.where(residue != 5, 1.0, 0.0)
                src_mask_repeat = src_mask.squeeze(1).repeat(sample_num,1)
                contact_prob = contact.repeat(sample_num, 1).masked_fill(src_mask_repeat.eq(0) | residue_mask.eq(0), 0)
                contact_prob    = torch.softmax(contact_prob * 5, dim=-1)
                contact_idx     = topkp_random(contact_prob, top_k=3, top_p=0.9)
                coords = coords.repeat(sample_num,1,1)
                src_mask2 = src_mask.repeat(sample_num, max_step, 1)
                
                captions = torch.zeros(sample_num, max_step).long().cuda()
                star = torch.zeros(sample_num, max_step).long().cuda()
                gt_coords = torch.zeros(sample_num, max_step, 3).long().cuda()
                root_coords = torch.zeros(sample_num, max_step - 1, 3).long().cuda()
                root_coords_n1 = root_coords.clone()
                root_coords_n2 = root_coords.clone()
                smi_map = torch.zeros(sample_num, max_step).long().cuda()
                smi_map_n1 = smi_map.clone()
                smi_map_n2 = smi_map.clone()
                last_sep = torch.zeros(sample_num).long().cuda()
                batch_ind = torch.arange(sample_num).long().cuda()
                ele_mask = torch.zeros(sample_num, max_step).long().cuda()

                captions[:, 0] = 1
                captions = captions.long()

                new_fea0_ = self.gt_fea1(captions)
                gt_coords[:,0] =  coords[batch_ind,contact_idx]
                gt_coords_emb0 = self.coords_emb_gt(gt_coords)
                gt_coords_emb = gt_coords_emb0.reshape(gt_coords_emb0.shape[0], gt_coords_emb0.shape[1], -1)
                new_fea0 = torch.cat([new_fea0_, gt_coords_emb], dim=-1)
                tgt_fea  = self.pos_encode(new_fea0)
                
                tgt_mask = subsequent_mask(max_step).long().cuda()
                memory = memory.repeat(sample_num, 1, 1)
                isTerminal = torch.ones(sample_num).cuda()
                batch_ind_total = torch.unsqueeze(torch.range(0, smi_map.shape[0]-1).cuda(), dim=-1).repeat((1, smi_map[:, 1:].shape[1])).long()

                for i in range(1, max_step - 1):
                    edge_vec_topo = self.relative_emb_topo(gt_coords.float() / 10.0, gt_coords.float() / 10.0)
                    dist_sca_topo = self.dist_emb_topo(gt_coords.float() / 10.0, gt_coords.float() / 10.0)
                    edge_fea_topo = torch.cat([edge_vec_topo, dist_sca_topo], dim=-1)
                    edge_fea_bias_topo = self.linears_edge1_topo(self.linears_edge_topo(edge_fea_topo)).permute(0, 3, 1, 2)


                    res, _ = self.decoder(tgt_fea, memory, src_mask2, tgt_mask, edge_fea_bias_topo)
                    proj_ = self.proj(res[:,i-1])
                    proj1 = F.softmax(proj_ , dim=-1)

                    if isMultiSample:
                        code = torch.multinomial(proj1,1,).view(-1)
                    else:
                        p_shape = proj1.shape
                        proj    = proj1.reshape(-1, p_shape[-1])
                        code    = proj.max(dim=-1)[1]

                    captions[:, i] = code  # 出拓扑结构  complete


                    isStar = torch.where((code==74)|(code==75),1.0,0.0)
                    star[:,i] = isStar
                    isSep_ind = torch.where(code == 3)
                    last_sep[isSep_ind] = i

                    '''
                    2) first cross att hidden  gt coords hidden with rooted ... get raw coords
                    '''

                    gt_c = gt_coords.clone()
                    gt_c[:, 0] = gt_c[:, 0] * 0
                    pre_idx, star, is_ele = find_root_smi_cur(captions, i, star)

                    ele_mask[:, i] = is_ele
                    root_coords[:, i - 1] = gt_c[batch_ind, pre_idx.long()]

                    smi_map[:, i] = pre_idx.long()

                    pre_idx_root = smi_map[batch_ind, pre_idx.long()]
                    pre_idx_root = pre_idx_root * is_ele + pre_idx * (1 - is_ele)

                    smi_map_n1[:, i] = pre_idx_root.long()
                    root_coords_n1[:, i - 1] = gt_c[batch_ind, pre_idx_root.long()]

                    pre_idx_root_root = smi_map[batch_ind, pre_idx_root.long()]
                    pre_idx_root_root = pre_idx_root_root * is_ele + pre_idx * (1 - is_ele)

                    smi_map_n2[:, i] = pre_idx_root_root.long()
                    root_coords_n2[:, i - 1] = gt_c[batch_ind, pre_idx_root_root.long()]
                    root_coords_temp = root_coords.clone()
                    root_coords_temp[:, 0] = coords[batch_ind, contact_idx]

                    gt_coords_emb_rooted = self.coords_emb_gt(root_coords_temp)
                    gt_coords_emb_rooted1 = gt_coords_emb_rooted.reshape(gt_coords_emb_rooted.shape[0],
                                                                         gt_coords_emb_rooted.shape[1], -1)

                    type_fea2 = self.gt_fea1(captions[:, 1:])

                    target_type = type_fea2
                    new_fea2 = torch.cat([target_type, gt_coords_emb_rooted1], dim=-1)
                    tgt_fea2 = self.pos_encode(new_fea2)
                    src_mask3 = src_mask.repeat(sample_num, captions.size(1) - 1, 1)


                    tgt_mask2 = subsequent_mask( self.tgt_size - 1).long().cuda()
                    edge_vec = self.relative_emb(root_coords.float() / 10.0, root_coords.float() / 10.0)
                    dist_sca = self.dist_emb(root_coords.float() / 10.0, root_coords.float() / 10.0)
                    edge_fea = torch.cat([edge_vec, dist_sca], dim=-1)
                    edge_fea_bias = self.linears_edge1(self.linears_edge(edge_fea)).permute(0, 3, 1, 2)
                    res_aux, _ = self.decoder_relative(tgt_fea2, memory, src_mask3, tgt_mask2, edge_fea_bias, None)



                    '''
                    3) second att with mapskeleton ... get more acc coords
                    '''


                    '''
                    dist theta degree
                    '''

                    root_idx = smi_map[:, 1:]
                    root_root_idx = smi_map_n1[:, 1:]
                    root_root_root_idx = smi_map_n2[:, 1:]

                    '''
                    features
                    '''
                    new_res  = res[:,:-1]

                    root_fea = new_res[batch_ind_total, root_idx]
                    root_root_fea = new_res[batch_ind_total, root_root_idx]
                    root_root_root_fea = new_res[batch_ind_total, root_root_root_idx]

                    feature_dist = torch.cat((target_type, new_res,root_fea), dim=-1)
                    dist_pred = F.softmax(self.proj2_aux(feature_dist), dim=-1)
                    dist1 = torch.max(dist_pred, dim=-1)[1]
                    dist = dist1[:, i - 1]

                    feature_theta = torch.cat((target_type, new_res, root_fea, root_root_fea),dim=-1)
                    theta_pred = F.softmax(self.proj3_aux(feature_theta), dim=-1)
                    theta1 = torch.max(theta_pred, dim=-1)[1]
                    theta = theta1[:, i - 1]

                    feature_degree = torch.cat((target_type, new_res, root_fea, root_root_fea,root_root_root_fea), dim=-1)
                    degree_pred = F.softmax(self.proj4_aux(feature_degree), dim=-1)
                    degree1 = torch.max(degree_pred, dim=-1)[1]
                    degree = degree1[:, i - 1]

                    '''
                    4) third att with gt ... get final acc coords
                    '''

                    pos_res = res_aux +tgt_fea2
                    pos_res1 = pos_res
                    x_p_hidden = self.fcx(pos_res1)
                    x_p = F.softmax(x_p_hidden, dim=-1)

                    x_p_feature = torch.cat([pos_res1, x_p_hidden], dim=-1)
                    y_p_hidden = self.fcy(x_p_feature)
                    y_p = F.softmax(y_p_hidden, dim=-1)

                    y_p_feature = torch.cat([x_p_feature, y_p_hidden], dim=-1)
                    z_p = F.softmax(self.fcz(y_p_feature), dim=-1)

                    gt_coords_pre, seg_coords = segment_coords(gt_coords, last_sep, ele_mask)

                    if i==1:

                        coords_pred = next_coords(captions=captions, is_ele=is_ele, center_=coords[batch_ind,contact_idx],
                                                  pre_idx=pre_idx,
                                                  gt_coords_=gt_coords_pre, seg_coords=seg_coords,
                                                  center_pre_=root_coords_n1[:, i - 1],
                                                  center_pre_pre_=root_coords_n2[:, i - 1], dist_=dist,
                                                  theta_=theta, degree_=degree, x_prod_=x_p[:, i - 1],
                                                  y_prod_=y_p[:, i - 1],
                                                  z_prod_=z_p[:, i - 1], seq_idx=i)
                    else:

                        coords_pred = next_coords(captions=captions, is_ele=is_ele, center_=root_coords[:, i - 1],
                                                  pre_idx=pre_idx,
                                                  gt_coords_=gt_coords_pre, seg_coords=seg_coords,
                                                  center_pre_=root_coords_n1[:, i - 1],
                                                  center_pre_pre_=root_coords_n2[:, i - 1], dist_=dist,
                                                  theta_=theta, degree_=degree, x_prod_=x_p[:, i - 1],
                                                  y_prod_=y_p[:, i - 1],
                                                  z_prod_=z_p[:, i - 1], seq_idx=i)

                    gt_coords[:, i] = coords_pred

                    gt_coords = gt_coords.long()

                    new_fea0_ = self.gt_fea1(captions)
                    # gt_coords_emb0 = self.coords_emb_gt(gt_coords)
                    gt_coords_emb0 = self.coords_emb_gt(gt_coords)

                    gt_coords_emb = gt_coords_emb0.reshape(gt_coords_emb0.shape[0], gt_coords_emb0.shape[1], -1)
                    new_fea0 = torch.cat([new_fea0_, gt_coords_emb], dim=-1)

                    tgt_fea = self.pos_encode(new_fea0)

                    terminal = torch.where((code==2) | (code==0),0,1)
                    isTerminal = terminal*isTerminal

                    if torch.sum(isTerminal)==0:
                        break


        return captions, gt_coords,


if __name__ == '__main__':

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    coords = torch.LongTensor(np.random.randint(0, 24, (2, 500, 3))).cuda()
    gtcoords = torch.LongTensor(np.random.randint(0, 240, (2, 100, 3))).cuda()
    gtcoords_label = torch.LongTensor(np.random.randint(0, 240, (2, 99, 3))).cuda()
    gtcoords_label_n1 = torch.LongTensor(np.random.randint(0, 240, (2, 99, 3))).cuda()
    atom_type = torch.LongTensor(np.random.randint(0, 12, (2, 500))).cuda()
    src_mask = torch.LongTensor(np.random.randint(0, 2, (2, 500))).cuda()
    residue = torch.LongTensor(np.random.randint(0, 2, (2, 500))).cuda()
    criticle_anchor = torch.LongTensor(np.random.randint(0, 1, (2, 500))).cuda()


    cap = torch.LongTensor(np.random.randint(0, 70, (2, 100))).cuda()
    contact = torch.FloatTensor(np.random.rand(2,500)).cuda()
    model = TransformerModel()
    model.cuda()
    res = model(coords,atom_type,residue,criticle_anchor,src_mask,isTrain=False,contact=contact)
    print(res)
