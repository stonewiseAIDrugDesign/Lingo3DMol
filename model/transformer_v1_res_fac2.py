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
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED
from multiprocessing import Pool
import threadpool
from model.transformer_v1_res_mp1 import subsequent_mask,topkp_random,find_root_smi_cur,segment_coords,next_coords
import matplotlib.pyplot as plt
import threading
    
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size=12, gt_vocab_size=76, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.src_vocab_size = 12 # 蛋白元素类型 ['C', 'N', 'O', 'S', 'F', 'Cl', 'Se', 'P','Ca','Zn','Centroid']

        self.cap_size = 100      # ligand 长度

        self.coords_size = 99    # ligand 长度-1

        self.h = 8               # 8个head

        self.coords_emb = nn.Embedding(240, 128) # 蛋白坐标编码 TODO 历史原因 训练 mapskeleton hopping 所以分开

        self.coords_emb_gt = nn.Embedding(240, 128) # ligand 坐标编码

        self.residue    = nn.Embedding(50,128*3)    # padding 0 普通原子 1 hba 2 hbd 3 if aromatic + 4 , aromatic centroid 5

        self.anchor_emb = nn.Embedding(20,128) # 0,1

        self.relative_emb = edge_vector()      # 相对坐标 向量 embedding 默认+-4A 0.1A resolution

        self.dist_emb     = cdist()            # 相对dist embedding 4A 0.1A resolution

        self.relative_emb_topo = edge_vector()  # 预测topo 相对坐标 向量 embedding

        self.dist_emb_topo     = cdist()        # 预测topo 相对坐标向量embeding

        self.src_fea = nn.Embedding(src_vocab_size, 128)
        self.gt_fea = nn.Embedding(gt_vocab_size, 128)
        self.gt_fea1 = nn.Embedding(gt_vocab_size, 128)



        self.linears_edge = MLP(in_features=64, hidden_layer_sizes=[64, 32], out_features=self.h, dropout_p=0.1)

        self.linears_edge1 = nn.Linear(self.h, self.h)

        self.linears_edge_topo = MLP(in_features=64, hidden_layer_sizes=[64, 32], out_features=self.h, dropout_p=0.1)

        self.linears_edge1_topo = nn.Linear(self.h, self.h)

        c = copy.deepcopy

        d_model = 512

        attn = MultiHeadedAttention(h=8, d_model=d_model)

        attn_att = MultiHeadedAttention_att(h=8, d_model=d_model)

        attnbias = MultiHeadedAttentionBias(h=8, d_model=d_model)

        ff = PositionwiseFeedForward(d_model=512, d_ff=2048, dropout=dropout)

        self.pos_encode = PositionalEncoding(d_model, dropout)
        self.encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        encoder_layer_num = 3
        self.encoder = Encoder(self.encoder_layer, encoder_layer_num)

        '''
        type decoder
        '''
        self.decoder_layer = DecoderLayerBias(d_model, c(attnbias), c(attn_att), c(ff), dropout)
        decoder_layer_num = 3
        self.decoder = DecoderBias(self.decoder_layer, decoder_layer_num)

        '''
        relative coords
        '''
        self.decoder_layerbias = DecoderLayerBias(d_model, c(attnbias), c(attn_att), c(ff), dropout)
        decoder_layer_num = 3

        self.decoder_relative = DecoderBias(self.decoder_layerbias, decoder_layer_num)


        # GPT 各个原子类型,坐标 gpt 每个原子的特征

        self.proj     = nn.Linear(d_model, gt_vocab_size)  # 类型
        self.proj_aux = nn.Linear(d_model, gt_vocab_size)

        self.proj2_aux = MLP(in_features=2 * d_model + 128, hidden_layer_sizes=[256] * 2, out_features=13, dropout_p=0.1)
        self.proj3_aux = MLP(in_features=3 * d_model + 128, hidden_layer_sizes=[256] * 2, out_features=181, dropout_p=0.1)
        self.proj4_aux = MLP(in_features=4 * d_model + 128, hidden_layer_sizes=[256] * 2, out_features=181, dropout_p=0.1)

        self.proj2 = MLP(in_features=2*d_model+128,hidden_layer_sizes=[256] * 2,out_features=13, dropout_p=0.1)
        self.proj3 = MLP(in_features=3*d_model+128,hidden_layer_sizes=[256] * 2,out_features=181, dropout_p=0.1)
        self.proj4 = MLP(in_features=4*d_model+128,hidden_layer_sizes=[256] * 2,out_features=181, dropout_p=0.1)

        self.fcx =  MLP(in_features=d_model,hidden_layer_sizes=[256] * 2,out_features=240, dropout_p=0.1)
        self.fcy = MLP(in_features=d_model+240,hidden_layer_sizes=[256] * 2,out_features=240, dropout_p=0.1)
        self.fcz = MLP(in_features=d_model+2*240,hidden_layer_sizes=[256] * 2,out_features=240, dropout_p=0.1)

        self.proj_matrix  = MLP(in_features= 8, hidden_layer_sizes=[32] * 2, out_features=11, dropout_p=0.1)
        self.proj_matrix1 = MLP(in_features= 8, hidden_layer_sizes=[32] * 2, out_features=11, dropout_p=0.1)
        self.proj_contact = nn.Sequential(nn.Linear(1024,512),nn.Linear(512,1),nn.Sigmoid())
        self.proj_residue = nn.Linear(1024,512)

    def forward(self, coords, type, residue,critical_anchor,src_mask, 
                captions=None,gt_coords=None, smi_map=None, smi_map_n1=None, 
                smi_map_n2=None,isTrain=True, contact_idx=None, sample_num=1, max_sample_step=100, 
                isMultiSample=False,USE_THRESHOLD=False,isGuideSample=False,guidepath=None,
                isDegreeSample=False,isDiverseSample=True,start_this_step=0,OnceMolGen=False,frag_len=0,tempture=1.0):

        coords_emb       = self.coords_emb(coords)#这里编码对应的都是蛋白的原子参数饿位置参数，因此对应的都是encoder部分

        coords_embedding = torch.reshape(coords_emb, (coords_emb.shape[0], coords_emb.shape[1], -1))#每个方向的坐标编码到128长度，三个方向合成一排就是128*3长度

        type_emb      = self.src_fea(type)#128长度

        total_feature = torch.cat((coords_embedding, type_emb), dim=-1)#合起来，一共512长度

        residue_emb   = self.residue(residue)#编码到128*3长度

        critical_anchor_emb = self.anchor_emb(critical_anchor)#编码到128长度

        residue_total_emb       = torch.cat([residue_emb,critical_anchor_emb],dim=-1)#合起来，一共512长度

        total_feature1 = total_feature+residue_total_emb

        final_fea = self.pos_encode(total_feature1)#pos——enc
        #print(src_mask) 这个就是表示句子长度的mask，估计就是前面有东西的部分是1，后面没有东西了的部分是0

        src_mask  = src_mask.unsqueeze(1)
        #before torch.Size([1, 500])
        #after torch.Size([1, 1, 500])
        #但是输入的句子长度是不一样的，计算attention score会出现偏差，为了保证句子的长度一样所以需要进行填充，但是用0填充的位置的信息是完全没有意义的（多余的），经过softmax操作也会有对应的输出，会影响全局概率值，因此我们希望这个位置不参与后期的反向传播过程。以此避免最后影响模型自身的效果，既在训练时将补全的位置给Mask掉

        src_mask1 = src_mask.repeat(1, src_mask.size(-1), 1)#这样就相当于把所有的为0的行全部mask掉了，也就相当于前面的行都是1，后面的行都是0的一个500*500的mask矩阵
        #final torch.Size([1, 500, 500])
        memory    = self.encoder(final_fea, src_mask1)
        
        with torch.no_grad():
            #print(sample_num) 发现结果为100，应该就是用这个模型生成100个tocken
            
            residue      = residue.repeat(sample_num, 1)#residue为[1,500]，应该就是蛋白质上每一个原子的residue，比如hba、hbi之类的参数
            #repeat：当参数只有两个时：（行的倍数，列的倍数），复制成100*500的形式。
            
            residue_mask    = torch.where(residue != 5, 1.0, 0.0)

            coords = coords.repeat(sample_num,1,1)#这个coords还是蛋白坐标，torch.Size([1, 500, 3])，表示了蛋白上每个点的坐标
            #重复后，变为torch.Size([100, 500, 3])
            
            #print(max_sample_step)，结果也是100
            
            #最大采样次数，猜测可能是采样多少次，之后从里面选一个好的出来还是啥？
            src_mask2 = src_mask.repeat(sample_num, max_sample_step, 1)#torch.Size([100, 100, 500]
            
            captions  = torch.zeros(src_mask2.shape[0], max_sample_step ).long().cuda()

            star      = torch.zeros(src_mask2.shape[0],max_sample_step).long().cuda()

            gt_coords = torch.zeros(src_mask2.shape[0], max_sample_step , 3).long().cuda()#sample_num估计就是抽样次数，可以抽样100此？
            
            root_coords    = torch.zeros(src_mask2.shape[0], max_sample_step - 1, 3).long().cuda()

            root_coords_n1 = torch.zeros(src_mask2.shape[0], max_sample_step - 1, 3).long().cuda()

            root_coords_n2 = torch.zeros(src_mask2.shape[0], max_sample_step - 1, 3).long().cuda()

            smi_map    = torch.zeros(src_mask2.shape[0], max_sample_step).long().cuda(src_mask2.device)

            smi_map_n1 = torch.zeros(src_mask2.shape[0], max_sample_step).long().cuda(src_mask2.device)

            smi_map_n2 = torch.zeros(src_mask2.shape[0], max_sample_step).long().cuda(src_mask2.device)

            last_sep   = torch.zeros(src_mask2.shape[0]).long().cuda()

            batch_ind  = torch.arange(0, src_mask2.shape[0]).long().cuda(src_mask2.device)
            
            ele_mask   = torch.zeros(src_mask2.shape[0], max_sample_step).long().cuda(src_mask2.device)

            captions[:, 0] = 1

            captions = captions.long()

            Value_prob = torch.zeros(src_mask2.shape[0], max_sample_step ).float().cuda()

            new_fea0_ = self.gt_fea1(captions)
            
            gt_coords[:,0] =  coords[batch_ind,contact_idx]

            gt_coords_emb0 = self.coords_emb_gt(gt_coords)

            gt_coords_emb = gt_coords_emb0.reshape(gt_coords_emb0.shape[0], gt_coords_emb0.shape[1], -1)#和之前一样，变成128*3

            new_fea0 = torch.cat([new_fea0_, gt_coords_emb], dim=-1)#拼起来，变成128*4的长度

            tgt_fea  = self.pos_encode(new_fea0)#ligand，生成的小分子的特征向量，做好posenc就可以送入decoder里面运算

            tgt_size = max_sample_step

            tgt_mask = subsequent_mask(tgt_size).long().cuda()

            memory = memory.repeat(sample_num, 1, 1)

            isTerminal = torch.ones(src_mask2.shape[0]).cuda()

            isValidNum = torch.zeros(src_mask2.shape[0]).cuda()
            #这样看来是一个长度为采样长度的2维数组，估计就是表示每个分子有没有结束吧，只要还没结束，还需要采样就设置为1，不然就设置为0停止采样了
            
            batch_ind_total = torch.unsqueeze(torch.range(0, smi_map.shape[0]-1).cuda(), dim=-1).repeat((1, smi_map[:, 1:].shape[1]))

            batch_ind_total = batch_ind_total.long()
            #开始生成分子
            for i in range(1, max_sample_step - 1):
                # import pdb;pdb.set_trace()
                if (guidepath[1][:,i,:]==-1).all(axis=-1).all(axis=-1) or i>=start_this_step: # 如果没有坐标信息则要走模型预测
                    print(i,'res go model')
                    edge_vec_topo = self.relative_emb_topo(gt_coords.float() / 10.0, gt_coords.float() / 10.0)

                    dist_sca_topo = self.dist_emb_topo(gt_coords.float() / 10.0, gt_coords.float() / 10.0)

                    edge_fea_topo = torch.cat([edge_vec_topo, dist_sca_topo], dim=-1)

                    edge_fea_bias_topo = self.linears_edge1_topo(self.linears_edge_topo(edge_fea_topo)).permute(0, 3, 1, 2)
                    #print(memory.device)
                    res, _ = self.decoder(tgt_fea, memory, src_mask2, tgt_mask, edge_fea_bias_topo)
                    proj_ = self.proj(res[:,i-1])#res是中间decoder层的输出，而proj应该是从decoder层到后面tocken预测层中间的线性层。用它来预测下一个tocken是什么
                    proj1 = F.softmax(proj_/tempture, dim=-1)
                if isGuideSample and i<start_this_step:
                    print(i,'code go guide')
                    code = torch.from_numpy(guidepath[0][:,i]).long().cuda()
                else:
                    print(i,'code go model')
                    if USE_THRESHOLD:
                        proj1[proj1<0.013] = 0
                    if isMultiSample:
                        code = torch.multinomial(proj1,1,).view(-1)
                    else:
                        p_shape = proj1.shape
                        proj    = proj1.reshape(-1, p_shape[-1])
                        code    = proj.max(dim=-1)[1]
                if i>=start_this_step:
                    Value_prob[:, i] = torch.gather(proj1.cuda(), 1, code.unsqueeze(1)).squeeze(1)
                captions[:, i] = code  
                isStar = torch.where((code==74)|(code==75),1.0,0.0)
                star[:,i] = isStar
                isSep_ind = torch.where(code == 3)
                last_sep[isSep_ind] = i
                '''
                2) first cross att hidden  gt coords hidden with rooted ... get raw coords
                '''
                #print(captions.device)
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

                if (guidepath[1][:,i,:]==-1).all(axis=-1).all(axis=-1) or i>=start_this_step: # 如果没有指引则走模型
                    print(i,'coords go model')
                    gt_coords_emb_rooted = self.coords_emb_gt(root_coords_temp)

                    gt_coords_emb_rooted1 = gt_coords_emb_rooted.reshape(gt_coords_emb_rooted.shape[0],
                                                                            gt_coords_emb_rooted.shape[1], -1)

                    type_fea2 = self.gt_fea1(captions[:, 1:])

                    target_type = type_fea2

                    new_fea2 = torch.cat([target_type, gt_coords_emb_rooted1], dim=-1)

                    tgt_fea2 = self.pos_encode(new_fea2)

                    src_mask3 = src_mask.repeat(sample_num, captions.size(1) - 1, 1)

                    tgt_size2 = self.cap_size - 1

                    tgt_mask2 = subsequent_mask(tgt_size2).long().cuda()

                    edge_vec = self.relative_emb(root_coords.float() / 10.0, root_coords.float() / 10.0)

                    dist_sca = self.dist_emb(root_coords.float() / 10.0, root_coords.float() / 10.0)

                    edge_fea = torch.cat([edge_vec, dist_sca], dim=-1)

                    edge_fea_bias = self.linears_edge1(self.linears_edge(edge_fea)).permute(0, 3, 1, 2)

                    res_aux, _ = self.decoder_relative(tgt_fea2, memory, src_mask3, tgt_mask2, edge_fea_bias, None)

                    #print(tgt_fea2.device)
                    #print(memory.device)
                    

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
                    # 
                    dist1 = torch.max(dist_pred, dim=-1)[1]
                    dist = dist1[:, i - 1]

                    feature_theta = torch.cat((target_type, new_res, root_fea, root_root_fea),dim=-1)
                    theta_pred = F.softmax(self.proj3_aux(feature_theta), dim=-1)
                    theta1 = torch.max(theta_pred, dim=-1)[1]
                    theta = theta1[:, i - 1]

                    feature_degree = torch.cat((target_type, new_res, root_fea, root_root_fea,root_root_root_fea), dim=-1)
                    degree_pred = F.softmax(self.proj4_aux(feature_degree), dim=-1)
                    if isDegreeSample:
                        # print(111111111111111)
                        degree1 = torch.multinomial(degree_pred.view(-1,degree_pred.size(-1)),1,).view(degree_pred.size(0),-1)
                    else:
                        degree1 = torch.max(degree_pred, dim=-1)[1]
                    # degree1 = torch.max(degree_pred, dim=-1)[1]
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
                    # cor_start=time.time()
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
                        # if i == 20:
                            # import pdb;pdb.set_trace()
                        coords_pred = next_coords(captions=captions, is_ele=is_ele, center_=root_coords[:, i - 1],
                                                    pre_idx=pre_idx,
                                                    gt_coords_=gt_coords_pre, seg_coords=seg_coords,
                                                    center_pre_=root_coords_n1[:, i - 1],
                                                    center_pre_pre_=root_coords_n2[:, i - 1], dist_=dist,
                                                    theta_=theta, degree_=degree, x_prod_=x_p[:, i - 1],
                                                    y_prod_=y_p[:, i - 1],
                                                    z_prod_=z_p[:, i - 1], seq_idx=i)
                elif isGuideSample and i<start_this_step: # 如果有指导坐标则进行赋值
                    print(i,'coords go gudie')
                    coords_pred = torch.from_numpy(guidepath[1][:,i]).long().cuda()
                
                # print('cor time is ',time.time()-cor_start)
                gt_coords[:, i] = coords_pred
                print(coords_pred[0])
                gt_coords = gt_coords.long()

                new_fea0_ = self.gt_fea1(captions)

                # gt_coords_emb0 = self.coords_emb_gt(gt_coords)
                gt_coords_emb0 = self.coords_emb_gt(gt_coords)

                gt_coords_emb = gt_coords_emb0.reshape(gt_coords_emb0.shape[0], gt_coords_emb0.shape[1], -1)
                new_fea0 = torch.cat([new_fea0_, gt_coords_emb], dim=-1)
                # print("new",new_fea0.shape,new_fea0)
                tgt_fea = self.pos_encode(new_fea0)
                if OnceMolGen:
                    if i>=start_this_step:
                        terminal = torch.where((code==2) | (code==0),0,1)
                        isTerminal = terminal*isTerminal
                        valid_index = torch.where(code==2)[0][isValidNum[torch.where(code==2)]==0]
                        isValidNum[valid_index] = i
                else:
                    if i>=start_this_step and i<=start_this_step+frag_len: # 观测盲区，只找2
                        terminal = torch.where((code==2) | (code==0),0,1)
                        isTerminal = terminal*isTerminal
                        valid_index = torch.where(code==2)[0][isValidNum[torch.where(code==2)]==0]
                        isValidNum[valid_index] = i
                    elif i>start_this_step+frag_len: # 观测区找3
                        terminal = torch.where((code==3) | (code==0),0,1)
                        isTerminal = terminal*isTerminal
                        valid_index = torch.where(code==3)[0][isValidNum[torch.where(code==3)]==0]
                        isValidNum[valid_index] = i

                if torch.sum(isTerminal)==0:
                    break
        isValidNum = isValidNum.cpu().numpy().astype(int)
        mask = (np.arange(100)<=isValidNum[:,np.newaxis]).astype(int)
        return captions, gt_coords,mask,Value_prob

    
if __name__ == '__main__':

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    coords = torch.LongTensor(np.random.randint(0, 24, (2, 500, 3))).cuda()

    gtcoords = torch.LongTensor(np.random.randint(0, 240, (2, 100, 3))).cuda()

    gtcoords_label = torch.LongTensor(np.random.randint(0, 240, (2, 99, 3))).cuda()

    gtcoords_label_n1 = torch.LongTensor(np.random.randint(0, 240, (2, 99, 3))).cuda()

    type = torch.LongTensor(np.random.randint(0, 12, (2, 500))).cuda()

    src_mask = torch.LongTensor(np.random.randint(0, 2, (2, 500))).cuda()

    residue = torch.LongTensor(np.random.randint(0, 2, (2, 500))).cuda()

    criticle_anchor = torch.LongTensor(np.random.randint(0, 1, (2, 500))).cuda()


    cap = torch.LongTensor(np.random.randint(0, 70, (2, 100))).cuda()

    contact = torch.FloatTensor(np.random.rand(2,500)).cuda()

    model = TransformerModel()

    model.cuda()

    res = model(coords,type,residue,criticle_anchor,src_mask,isTrain=False,contact=contact)

    print(res)
