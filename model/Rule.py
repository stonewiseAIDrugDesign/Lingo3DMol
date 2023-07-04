# This file is part of Lingo3DMol
#
# Lingo3DMol is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Lingo3DMol is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Lingo3DMol. If not, see <https://www.gnu.org/licenses/>.



import numpy as np
import torch
import torch.nn as nn
import time
import copy
from copy import deepcopy
import math
import torch.nn.functional as F

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
    cube_template2 = cube(35)

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
    maskA1 = torch.where(d_c<=20,0.0,1.0)
    maskA2 = torch.where(d_c >= 35, 0.0, 1.0)
    mask1A = maskA1*maskA2


    joint_prob = joint_prob.masked_fill(mask1A == 0, -1e9)
    new_joint_prob = joint_prob
    new_joint_prob2 = torch.softmax(new_joint_prob, dim=-1)
    ind = torch.where(new_joint_prob2 == -1e9, 0.0, 1.0)
    idx = torch.max(new_joint_prob2, dim=-1)[1]
    indices_batch = torch.arange(0, len(center)).cuda().long()
    pred_coords = translate[indices_batch, idx]

    return pred_coords






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



def next_coords(captions,is_ele,center_,pre_idx,gt_coords_,seg_coords,center_pre_,center_pre_pre_,dist_,theta_,degree_,x_prod_,y_prod_,z_prod_,seq_idx):
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
        pred_coords_rule = make_cube_first(center_,x_prod_, y_prod_, z_prod_)
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

def topkp_random(logits,top_k=2,top_p=0.8,filter_value=-1e9,thred= 0.0):
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
