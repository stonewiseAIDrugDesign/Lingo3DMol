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


def cube(radius=21):
    cube_ = np.zeros((radius*2+1,radius*2+1,radius*2+1,3))
    for x in range(0, radius*2+1):
        for y in range(0, radius*2+1):
            for z in range(0, radius*2+1):
                cube_[x,y,z,0] = x
                cube_[x,y,z,1] = y
                cube_[x,y,z,2] =z
    cube_ = cube_.reshape(1,-1,3)
    return torch.FloatTensor(cube_)

cube_template1 = cube()

def make_cube(center,gt_coords,seg_coords,center_pre,center_pre_pre,dist,theta,degree,x_prod,y_prod,z_prod,isdist,istheta,isdegree,radius=21):
    # 2.1A 范围内的点
    cube_template=cube_template1.cuda(device=center.device)
    new_cubes = cube_template.repeat(len(center),1,1) # batch radius*2+1^3 ,3
    new_center = torch.unsqueeze(center,dim=1) # batch 1 3
    translate = new_cubes+new_center-21
    translate = torch.clamp(translate,min=0.0,max=239.0)
    # get joint prob
    indices   = torch.unsqueeze(torch.arange(0,len(center)),dim=-1).cuda() # batch 1 1
    indices2  = indices.repeat(1,(radius*2+1)*(radius*2+1)*(radius*2+1))  #
    joint_prob_x = x_prod[indices2,translate[:,:,0].long()]
    joint_prob_y = y_prod[indices2,translate[:,:,1].long()]
    joint_prob_z = z_prod[indices2,translate[:,:,2].long()]
    joint_prob = joint_prob_x*joint_prob_y*joint_prob_z
    # dist_to center
    d1  = translate - new_center
    d_c = torch.sqrt(torch.sum(torch.square(d1), dim=-1)) # batch   radius*2+1^3
    # mask 1A 范围内的点
    maskA1 = torch.where(d_c<=10,0.0,1.0)
    maskA2 = torch.where(d_c >= 21, 0.0, 1.0)
    mask1A = maskA1*maskA2
    joint_prob = joint_prob.masked_fill(mask1A == 0, -1e9)
    # mask 和 已经生成的点1.5A （14） 范围的点 （O 1.4）
    dist_pre = torch.cdist(translate,gt_coords.float(),p=2.0)
    dist_pre1 = torch.where(dist_pre<=14,0.0,1.0)
    dist_pre2 = torch.sum(dist_pre1,dim=-1)
    dist_pre_mask = torch.where(dist_pre2==gt_coords.shape[1],1.0,0.0)
    dist_pre_seg = torch.cdist(translate, seg_coords.float(), p=2.0)
    dist_pre1_seg = torch.where(dist_pre_seg <= 10, 0.0, 1.0)
    dist_pre2_seg = torch.sum(dist_pre1_seg, dim=-1)
    dist_pre_mask_seg = torch.where(dist_pre2_seg == seg_coords.shape[1], 1.0, 0.0)
    total_mask = mask1A* dist_pre_mask*dist_pre_mask_seg
    # mask 和已经生成的点

    # 取得 符合dist的点
    # if isdist:
    t1 = time.time()
    dist_new =  torch.unsqueeze(dist,dim=-1) # batch 1
    dist_new2 = dist_new.repeat(1,(radius*2+1)*(radius*2+1)*(radius*2+1))
    dist_new2_mask = torch.where(dist_new2==0,0.0,1.0) #记住等于0 的特殊位置
    dist_new3 = (dist_new2+9.0)*dist_new2_mask
    dist_new4 = torch.where((d_c<=dist_new3+1) & (d_c>=dist_new3-1),1.0,0.0) # orig 1 TODO
    dist_new4  = dist_new4 * isdist   # 会全部置为 0
    temp_mask = total_mask*dist_new4
    temp_mask_sig = torch.unsqueeze(torch.where(torch.sum(temp_mask,dim=-1)==0,0.0,1.0),dim=-1)
    total_mask = (1-temp_mask_sig)*total_mask + temp_mask_sig * temp_mask
    t2 = time.time()

    #获得指定角度的点
    pre_vec = (center_pre.float() - center.float()) / torch.unsqueeze(torch.linalg.norm((center_pre - center).float(),dim=-1) ,dim=-1)# batch 3
    pre_vec1 = torch.unsqueeze(pre_vec,dim=1) # batch 1 3
    pre_vec2 = pre_vec1.repeat(1,(radius*2+1)*(radius*2+1)*(radius*2+1),1) # batch ... 3
    cur_vec0 = (translate - new_center)
    cur_vec = cur_vec0/torch.unsqueeze(torch.linalg.norm(cur_vec0,dim=-1),dim=-1) # batch ... 3
    cos_theta = torch.arccos(torch.sum(cur_vec*pre_vec2,dim=-1))
    theta_cur = cos_theta*180/math.pi        #
    theta1 = torch.unsqueeze(theta,dim=-1)
    theta2 = theta1.repeat(1,(radius*2+1)*(radius*2+1)*(radius*2+1))
    theta_mask = torch.where((theta_cur<=(theta2+2.0)) & (theta_cur>=(theta2-2.0)),1.0,0.0)  # ToDo pre 2
    theta_mask = theta_mask * istheta  # 会全部置为 0
    temp_mask =  total_mask * theta_mask
    temp_mask_sig = torch.unsqueeze(torch.where(torch.sum(temp_mask, dim=-1) == 0, 0.0, 1.0),dim=-1)
    total_mask = (1 - temp_mask_sig) * total_mask + temp_mask_sig * temp_mask
    t3 = time.time()
    # print('angle', t3-t2)
    # if isdegree:
    c1 = (center_pre.float()-center.float())/torch.unsqueeze(torch.linalg.norm((center_pre - center).float(),dim=-1),dim=-1)
    c2 = (center_pre_pre.float()-center.float())/torch.unsqueeze(torch.linalg.norm((center_pre_pre.float()-center.float()),dim=-1),dim=-1)
    v1_plane  = torch.cross(c1,c2)
    v1_plane = v1_plane/torch.unsqueeze(torch.linalg.norm(v1_plane,dim=-1),dim=-1)
    v1_plane0 = torch.unsqueeze(v1_plane,dim=1)
    v1_plane1 = v1_plane0.repeat(1, (radius * 2 + 1) * (radius * 2 + 1) * (radius * 2 + 1), 1)
    t5 = time.time()
    # print('v1 plane',t5-t3)
    pre_vec1   = torch.unsqueeze(c1, dim=1)  # batch 1 3
    pre_vec2   = pre_vec1.repeat(1, (radius * 2 + 1) * (radius * 2 + 1) * (radius * 2 + 1), 1)
    cur_vec0   = (translate - new_center)
    cur_vec    = cur_vec0/torch.unsqueeze(torch.linalg.norm(cur_vec0,dim=-1),dim=-1)
    v2_plane   = torch.cross(pre_vec2,cur_vec)
    v2_plane   = v2_plane/torch.unsqueeze(torch.linalg.norm(v2_plane,dim=-1),dim=-1)
    t7 = time.time()
    # print('v2 plane',t7-t5)
    cos_degree = torch.arccos(torch.sum(v1_plane1 * v2_plane, dim=-1))
    degree_cur   = cos_degree * 180 / math.pi
    # c = degree_cur.cpu().numpy()
    t6 = time.time()
    new_degree = degree#torch.where(degree<90,0,1)*90
    degree1 = torch.unsqueeze(new_degree, dim=-1)
    degree2 = degree1.repeat(1, (radius * 2 + 1) * (radius * 2 + 1) * (radius * 2 + 1))
    degree_mask  = torch.where((degree_cur<=(degree2.float()+2.0)) & (degree_cur>=(degree2.float()-2.0)),1.0,0.0) #之前这里写的是5
    temp_mask = total_mask*degree_mask
    degree_mask = degree_mask * isdegree  # 会全部置为 0
    temp_mask = total_mask * degree_mask
    temp_mask_sig = torch.unsqueeze(torch.where(torch.sum(temp_mask, dim=-1) == 0, 0.0, 1.0),dim=-1)
    total_mask = (1 - temp_mask_sig) * total_mask + temp_mask_sig * temp_mask
    t4 = time.time()
    total_mask1 = mask1A * dist_pre_mask
    temp_mask_sig = torch.unsqueeze(torch.where(torch.sum(total_mask, dim=-1) == 0, 0.0, 1.0),dim=-1)
    total_mask == (1-temp_mask_sig)*total_mask1 + temp_mask_sig * total_mask
    new_joint_prob = joint_prob.masked_fill(total_mask == 0, -1e9)
    new_joint_prob2 = torch.softmax(new_joint_prob,dim=-1)
    ind = torch.where(new_joint_prob2==-1e9,0.0,1.0)
    # print(torch.sum(ind))
    idx = torch.max(new_joint_prob2,dim=-1)[1]
    indices_batch = torch.arange(0,len(center)).cuda().long()
    pred_coords = translate[indices_batch,idx]

    return pred_coords



def make_cube_first(center,x_prod,y_prod,z_prod,radius=35):

    # 2.1A 范围内的点
    # if radius<=35:
    #     lower = 20
    # else:
    #     lower = 35
    lower =20
    cube_template2 = cube(radius)

    cube_template=cube_template2.cuda(device=center.device)
    new_cubes = cube_template.repeat(len(center),1,1) # batch radius*2+1^3 ,3
    new_center = torch.unsqueeze(center,dim=1) # batch 1 3
    translate = new_cubes+new_center-radius
    translate = torch.clamp(translate,min=0.0,max=239.0)
    # get joint prob
    indices   = torch.unsqueeze(torch.arange(0,len(center)),dim=-1).cuda() # batch 1 1
    indices2  = indices.repeat(1,(radius*2+1)*(radius*2+1)*(radius*2+1))  #
    joint_prob_x = x_prod[indices2,translate[:,:,0].long()]
    joint_prob_y = y_prod[indices2,translate[:,:,1].long()]
    joint_prob_z = z_prod[indices2,translate[:,:,2].long()]
    joint_prob = joint_prob_x*joint_prob_y*joint_prob_z
    # dist_to center
    d1  = translate - new_center
    d_c = torch.sqrt(torch.sum(torch.square(d1), dim=-1)) # batch   radius*2+1^3
    # mask 1A 范围内的点
    maskA1 = torch.where(d_c<=lower,0.0,1.0)
    maskA2 = torch.where(d_c >= radius, 0.0, 1.0)
    mask1A = maskA1*maskA2


    joint_prob = joint_prob.masked_fill(mask1A == 0, -1e9)
    new_joint_prob = joint_prob
    new_joint_prob2 = torch.softmax(new_joint_prob, dim=-1)
    ind = torch.where(new_joint_prob2 == -1e9, 0.0, 1.0)
    idx = torch.max(new_joint_prob2, dim=-1)[1]
    indices_batch = torch.arange(0, len(center)).cuda().long()
    pred_coords = translate[indices_batch, idx]

    return pred_coords



def clones(module, N):
    "Produce N identical layers."

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderBias(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(DecoderBias, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask,bias,crossbias=None):
        for layer in self.layers:
            x,bias,att = layer(x, memory, src_mask, tgt_mask,bias,crossbias)
        return self.norm(x),att

class DecoderGPT(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(DecoderGPT, self).__init__()
        self.norm = LayerNorm(layer.size)
        self.gpts = clones(layer, N)

    def forward(self, x, tgt_mask):
        for layer in self.gpts:
            x = layer(x, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class DecoderLayerBias(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayerBias, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 1)
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask,bias,crossbias=None):
        "Follow Figure 1 (right) for connections."
        m = memory
        x1,new_bias =  self.self_attn(x, x, x, bias,tgt_mask)
        x2 = x+self.dropout(self.norm1(x1))

        # if crossbias is not None:
        #     x3, cross_bias = self.src_attn(x2, m, m, crossbias, src_mask)
        # else:
        x3,att = self.src_attn(x2, m, m, src_mask)
        cross_bias = att
        x4 = x2+self.dropout(self.norm2(x3))
        return self.sublayer[0](x4, self.feed_forward),new_bias,cross_bias

class DecoderLayerGPT(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayerGPT, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def subsequent_mask1(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size+1)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # if mask is not None:
    scores1 = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores1, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), scores

def attentionbias( value, bias,mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # scores = torch.matmul(query, key.transpose(-2, -1)) \
    #          / math.sqrt(d_k)
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        bias1  = bias.masked_fill(mask==0,-1e9)
    weight = bias1
    p_attn = F.softmax(weight, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value),bias

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)


        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]



        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class MultiHeadedAttention_att(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention_att, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)


        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]



        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x),self.attn


class MultiHeadedAttentionBias(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionBias, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn          = None
        self.dropout       = nn.Dropout(p=dropout)

        # self.w             = nn.parameter.Parameter(torch.ones(self.h))
        # self.w1            = nn.parameter.Parameter(torch.FloatTensor([0.05]))
        # self.proj          = MLP(in_features=self.h,hidden_layer_sizes=[256,256],out_features=self.h,dropout_p=0.1)
    def forward(self, query, key, value,bias, mask=None):

        "Implements Figure 2"

        # w = self.w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        value = self.linears[0](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for l, x in zip(self.linears, ( value))]


        # edge_fea1 = self.w1*bias#bias.permute(0,3,1,2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attentionbias(value,bias=bias, mask=mask, dropout=self.dropout)

        # edge_fea2 = bias+w*self.attn

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)




        return self.linears[-1](x),bias



class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x1 = x + self.pe[:, :x.size(1)]
        return self.dropout(x1)



def find_in_other_frag(sample,j,star_single):
    # 找到 第几个* 号  map 和 sep map

    ele_token = [i for i in range(4, 56)]
    bracket_pre = [70]
    bracket_post = [71]

    res = torch.FloatTensor([0]).cuda()

    try:  # 以防idx 是 空的
        idx_star = torch.where(star_single[:j] == 1.0)[-1][-1]

        j = idx_star - 1

        while j >= 0:
            # 不可能再跨了
            if sample[j] in ele_token:
                res = j
                break

            if sample[j] in bracket_post:
                branch = []
                branch.append(deepcopy(j))

                k = j - 1
                f = False
                while k >= 0 and len(branch) > 0:
                    if sample[k] in bracket_post:
                        branch.append(deepcopy(k))
                        k -= 1
                        continue
                    if sample[k] in bracket_pre:
                        token_b = branch.pop(-1)
                        if sample[k] + 1 != sample[token_b]:
                            break
                        if sample[k] + 1 == sample[token_b] and len(branch) == 0:
                            f = True
                            break

                    k -= 1

                if f and k - 1 >= 0:
                    j = k - 1
                    continue
                else:
                    break

            j = j - 1
    except:
        return res,0

    return res,idx_star






def find_root_smi_cur(batch_codes,idx,star):
    ele_token = [i for i in range(4,56)]
    sep = [3]
    bracket_pre = [70]
    bracket_post = [71]
    pad =[0]
    is_ele = torch.zeros(len(batch_codes)).cuda()

    res  = torch.zeros(len(batch_codes)).cuda()

    for i, sample in enumerate(batch_codes):  # sample

            if idx==0:
                continue
            if sample[idx] in pad:
                continue

            if sample[idx] not in ele_token:
                j = idx

            else:
                is_ele[i] = 1
                j =  idx-1




            while j>=0:
                if sample[j] in  sep:
                    temp,idx_star = find_in_other_frag(sample,j,star[i]) # 第i 条 star 记录
                    # print(temp)
                    res[i] = deepcopy(temp)
                    if sample[idx] in ele_token:
                        star[i,idx_star]=2.0
                    break
                if sample[j] in ele_token:
                    res[i] = j
                    break

                if sample[j] in bracket_post:
                    branch = []
                    branch.append(deepcopy(j))

                    k  = j-1
                    f = False
                    while k>=0 and len(branch)>0:
                        if sample[k] in bracket_post:
                            branch.append(deepcopy(k))
                            k-=1
                            continue
                        if sample[k] in bracket_pre:

                            token_b = branch.pop(-1)
                            if sample[k]+1 !=sample[token_b]:
                                print('grammer err !',k,token_b,sample[k],sample[token_b],branch)
                                break
                            if sample[k] +1 == sample[token_b] and len(branch)==0 :
                                f = True
                                break

                        k-=1

                    if f and k-1>=0:
                        j = k-1
                        continue
                    else:
                        break

                j = j-1
    # print(idx,batch_codes,res,star,is_ele)

    return res,star,is_ele.long()







class MLP(torch.nn.Module):
    """
    Multi-layer perceptron. Applies SELU after every linear layer.

    Args:
    ----
        in_features (int)         : Size of each input sample.
        hidden_layer_sizes (list) : Hidden layer sizes.
        out_features (int)        : Size of each output sample.
        dropout_p (float)         : Probability of dropping a weight.
    """

    def __init__(self, in_features : int, hidden_layer_sizes : list, out_features : int,
                 dropout_p : float) -> None:
        super().__init__()

        activation_function = torch.nn.SELU

        # create list of all layer feature sizes
        fs = [in_features, *hidden_layer_sizes, out_features]

        # create list of linear_blocks
        layers = [self._linear_block(in_f, out_f,
                                     activation_function,
                                     dropout_p)
                  for in_f, out_f in zip(fs, fs[1:])]

        # concatenate modules in all sequentials in layers list
        layers = [module for sq in layers for module in sq.children()]

        # add modules to sequential container
        self.seq = torch.nn.Sequential(*layers)

    def _linear_block(self, in_f : int, out_f : int, activation : torch.nn.Module,
                      dropout_p : float) -> torch.nn.Sequential:
        """
        Returns a linear block consisting of a linear layer, an activation function
        (SELU), and dropout (optional) stack.

        Args:
        ----
            in_f (int)                   : Size of each input sample.
            out_f (int)                  : Size of each output sample.
            activation (torch.nn.Module) : Activation function.
            dropout_p (float)            : Probability of dropping a weight.

        Returns:
        -------
            torch.nn.Sequential : The linear block.
        """
        # bias must be used in most MLPs in our models to learn from empty graphs
        linear = torch.nn.Linear(in_f, out_f, bias=True)
        torch.nn.init.xavier_uniform_(linear.weight)
        return torch.nn.Sequential(linear, activation()) #, torch.nn.AlphaDropout(dropout_p)

    def forward(self, layers_input : torch.nn.Sequential) -> torch.nn.Sequential:
        """
        Defines forward pass.
        """
        return self.seq(layers_input)


def next_coords(captions,is_ele,center_,pre_idx,gt_coords_,seg_coords,center_pre_,center_pre_pre_,dist_,theta_,degree_,x_prod_,y_prod_,z_prod_,seq_idx,radius=35):
    t1 = time.time()
    gt_coords_post = copy.deepcopy(gt_coords_)


    gt_coords_post[torch.arange(len(gt_coords_post)).long().cuda(), pre_idx.long()] = gt_coords_post[torch.arange(len(gt_coords_post)).long().cuda(), pre_idx.long()] * 0
    seg_coords[torch.arange(len(seg_coords)).long().cuda(), pre_idx.long()] = seg_coords[torch.arange(len(seg_coords)).long().cuda(), pre_idx.long()] * 0

    dist_mask = torch.where(torch.sum(center_,dim=-1)==0,0.0,1.0)
    theta_mask = torch.where(torch.sum(center_pre_,dim=-1)==0,0.0,1.0)
    theta_mask = theta_mask*dist_mask
    degree_mask = torch.where(torch.sum(center_pre_pre_,dim=-1)==0,0.0,1.0)
    degree_mask = degree_mask*theta_mask

    dist_mask = torch.unsqueeze(dist_mask,dim=-1)
    theta_mask = torch.unsqueeze(theta_mask,dim=-1)
    degree_mask = torch.unsqueeze(degree_mask,dim=-1)

    x_ = x_prod_.max(dim=-1)[1]

    y_ = y_prod_.max(dim=-1)[1]
    #
    z_ = z_prod_.max(dim=-1)[1]
    #
    max_coords = torch.stack([x_,y_,z_],dim=-1)

    if seq_idx>1:
        pred_coords_rule = make_cube(center_, gt_coords_post,seg_coords ,center_pre_, center_pre_pre_,dist_,theta_, degree_,x_prod_, y_prod_, z_prod_, isdist=dist_mask,istheta=theta_mask,isdegree=degree_mask)
        pred_coords_total = (1 - dist_mask) * max_coords + dist_mask * pred_coords_rule

    else:
        pred_coords_rule = make_cube_first(center_,x_prod_, y_prod_, z_prod_,radius=radius)
        pred_coords_total = pred_coords_rule

    is_ele1 = torch.unsqueeze(is_ele,dim=1)
    pred_coords_total1 = pred_coords_total*is_ele1 + center_*(1-is_ele1)

    return pred_coords_total1


def segment_coords(gt_coords,last_sep,ele_mask):
    mask = torch.zeros_like(gt_coords).cuda()
    rmask = torch.ones_like(gt_coords).cuda()
    for i in range(len(last_sep)):
        mask[i,:last_sep[i],:]=1
        rmask[i,:last_sep[i],:]=0

    pre = mask*gt_coords*ele_mask.unsqueeze(-1)
    seg = rmask*gt_coords*ele_mask.unsqueeze(-1)
    return pre,seg

class cdist(nn.Module):

    def __init__(self,dist_num=42,embedding_dim=16):
        super(cdist,self).__init__()

        '''
        :param dist_num: 0.1A
        '''
        self.dist = nn.Embedding(dist_num,embedding_dim)

    def forward(self,input1,input2,emb_dim=3):
        input_shape1 = input1.shape
        input_shape2 = input2.shape

        # Flatten input
        flat_input1 = input1.reshape(input1.shape[0],-1, emb_dim)
        flat_input2 = input2.reshape(input2.shape[0],-1,emb_dim)
        flat_input3 = flat_input2.permute(0,2,1)


        # Calculate distances
        a= torch.sum(flat_input1 ** 2, dim=-1, keepdim=True)
        b = torch.sum(flat_input2 ** 2, dim=-1).unsqueeze(dim=1)
        c = torch.matmul(flat_input1, flat_input3)
        distances = torch.sqrt(a+b-2*c)

        dist_map = torch.clamp(distances.reshape(-1,input_shape1[1],input_shape2[1]),min=0,max=4)*10

        dist_map1 = torch.round(dist_map).long()

        dist      = self.dist(dist_map1)

        return dist


class edge_vector(nn.Module):
    def __init__(self,dist_num=85,embedding_dim=16):
        super(edge_vector,self).__init__()


        self.relative_coords = nn.Embedding(dist_num,embedding_dim)
    def forward(self,input1,input2):

        input1_new = input1.unsqueeze(dim=-2)

        input2_new = input2.unsqueeze(dim=1)

        new_vec = torch.round((torch.clamp(input1_new-input2_new,min=-4.0,max=4.0)+4.0)*10.0).long()

        new_vec1 = self.relative_coords(new_vec)

        new_vec2 = new_vec1.reshape(input1.shape[0],input1.shape[1],input2.shape[1],-1)

        return new_vec2



def segment_coords(gt_coords,last_sep,ele_mask):
    mask = torch.zeros_like(gt_coords).cuda()
    rmask = torch.ones_like(gt_coords).cuda()
    for i in range(len(last_sep)):
        mask[i,:last_sep[i],:]=1
        rmask[i,:last_sep[i],:]=0

    pre = mask*gt_coords*ele_mask.unsqueeze(-1)
    seg = rmask*gt_coords*ele_mask.unsqueeze(-1)
    return pre,seg

def segment_mask(gt_coords,last_sep,ele_mask,coords,thred=1.5):
    mask = torch.zeros_like(gt_coords).cuda()
    rmask = torch.ones_like(gt_coords).cuda()
    for i in range(len(last_sep)):
        mask[i, :last_sep[i], :] = 1
        rmask[i, :last_sep[i], :] = 0

    pre = mask * gt_coords * ele_mask.unsqueeze(-1)

    dist = torch.sqrt(torch.sum(torch.square(pre[:,:,None,:]/10.0-coords[:,None,...]/10.0),dim=-1))

    dist_mask = torch.where(dist<=thred,0,1) # B N M

    return dist_mask

def segment_interstarmask(pre_coords,coords,thred=1.5):

    dist = torch.sqrt(torch.sum(torch.square(pre_coords[:,None,:]/10.0-coords/10.0),dim=-1))
    dist_min = torch.min(dist,dim=-1)[0]
    dist_mask = torch.where(dist_min<=thred,1,0)
    return dist_mask

def topkp_random(logits,top_k=2,top_p=0.8,filter_value=-1e9,thred= 0.15):
    # 2 0.6

    top_k = min(top_k, logits.size(-1))

    if top_k > 0:

        topk = torch.topk(logits,top_k)[0][:,-1] # batchN

        topk = topk.reshape(-1,1)

        indices_to_remove = torch.where(logits < topk,0,1)

        logits = logits.masked_fill(indices_to_remove == 0, filter_value)

    if top_p > 0.0:

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs >= top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

        sorted_indices_to_remove[..., 0] = 0

        sorted_indices_to_remove = sorted_indices_to_remove.long()

        logits_zero = torch.zeros_like(logits,dtype=torch.long)

        indices_to_remove = logits_zero.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        logits = logits.masked_fill(indices_to_remove == 1, filter_value)

    logits_mask = torch.where(logits>thred,1,0)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    code = torch.zeros(logits_mask.shape[0]).cuda()

    for i in range(logits_mask.shape[0]):
        k = max(torch.sum(logits_mask[i]).item(),1)
        new_prob = sorted_logits[i,:k]
        new_probability = torch.softmax(new_prob/2,dim=-1)
        new_idx  = sorted_indices[i,:k]
        code_idx = torch.multinomial(new_probability, 1).view(-1)
        idx_code = new_idx[code_idx]
        code[i] = idx_code
        # idx = torch.randint(0, k, (1,))[0]
        # idx_code = sorted_indices[i, idx]
        # code[i] = idx_code
        #

    code = code.long()

    return code
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size=12, gt_vocab_size=76, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.src_vocab_size = 12

        self.cap_size = 100

        self.coords_size = 99

        self.h = 8

        self.coords_emb = nn.Embedding(240, 128)

        self.coords_emb_gt = nn.Embedding(240, 128)

        self.residue    = nn.Embedding(50,512)

        self.relative_emb = edge_vector()

        self.dist_emb     = cdist()

        self.relative_emb_topo = edge_vector()

        self.dist_emb_topo     = cdist()

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

        # '''
        # abs count
        # '''
        #
        # decoder_layer_num        = 1
        # self.decoder_layer2      = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)  #
        # self.decoder_mapskeleton = Decoder(self.decoder_layer2, decoder_layer_num)

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
        self.proj_contact_fea = nn.Sequential(nn.Linear(1024,512),nn.ReLU())
        self.proj_contact     = nn.Sequential(nn.Linear(512,1),nn.Sigmoid())
        self.proj_contact_scaffold = nn.Sequential(nn.Linear(512,1),nn.Sigmoid())
        self.proj_residue = nn.Linear(1024,512)

    def forward(self, coords, atom_type, residue,src_mask, captions=None, gt_coords=None, smi_map=None, smi_map_n1=None, smi_map_n2=None,isTrain=True, sample_num=1, max_sample_step=100):

        '''
        :param x:         density [N 1 48 48 48]
        :param coords:    [N,seq len, 3]
        :param atom_type:      [N,seq len, 1]
        :param captions:  [N,seq lens]
        :param isTrain:   bool
        :param sample_num:
        :param max_sample_step:
        :return:
        '''


        coords_emb       = self.coords_emb(coords)

        coords_embedding = torch.reshape(coords_emb, (coords_emb.shape[0], coords_emb.shape[1], -1))

        type_emb      = self.src_fea(atom_type)

        total_feature = torch.cat((coords_embedding, type_emb), dim=-1)


        residue_emb   = self.residue(residue)

        total_feature1 = total_feature+residue_emb

        final_fea = self.pos_encode(total_feature1)

        src_mask  = src_mask.unsqueeze(1)

        src_mask1 = src_mask.repeat(1, src_mask.size(-1), 1)

        memory    = self.encoder(final_fea, src_mask1)

        mem = memory

        contact_fea = torch.cat([mem,residue_emb],dim=-1)

        contact_fea0   = self.proj_contact_fea(contact_fea)

        contact    = self.proj_contact(contact_fea0).reshape([memory.shape[0],memory.shape[1]])

        contact_scaffold = self.proj_contact_scaffold(contact_fea0).reshape([memory.shape[0],memory.shape[1]])

        return contact,contact_scaffold


if __name__ == '__main__':

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    coords = torch.LongTensor(np.random.randint(0, 24, (2, 50, 3))).cuda()

    gtcoords = torch.LongTensor(np.random.randint(0, 240, (2, 100, 3))).cuda()

    gtcoords_label = torch.LongTensor(np.random.randint(0, 240, (2, 99, 3))).cuda()

    gtcoords_label_n1 = torch.LongTensor(np.random.randint(0, 240, (2, 99, 3))).cuda()

    atom_type = torch.LongTensor(np.random.randint(0, 12, (2, 50))).cuda()

    src_mask = torch.LongTensor(np.random.randint(0, 2, (2, 50))).cuda()

    residue = torch.LongTensor(np.random.randint(0, 2, (2, 50))).cuda()


    cap = torch.LongTensor(np.random.randint(0, 70, (2, 100))).cuda()

    model = TransformerModel()

    model.cuda()

    res = model(coords,atom_type,residue,src_mask,isTrain=False)

    print(res)