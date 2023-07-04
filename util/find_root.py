# This file is part of Lingo3DMol
#
# Lingo3DMol is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Lingo3DMol is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Lingo3DMol. If not, see <https://www.gnu.org/licenses/>.



import copy
from copy import deepcopy
import numpy as np
from rdkit import Chem
# from util.fragmol_frag_v0 import FragmolUtil
def find_in_other_frag(sample,j,star_single):
    # 找到 第几个* 号  map 和 sep map

    ele_token = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]#[i for i in range(4,56)]
    bracket_pre = [30]#[70]
    bracket_post = [31]#[71]
    res = 0

    try:  # 以防idx 是 空的
        idx_star = np.where(star_single[:j] == 1.0)[-1][-1]

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
                            print('grammer err ', sample[k], sample[token_b], branch)
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






def find_root_smi_cur(sample,idx,star):
    '''
    因为是给 非元素token 找的
    :param sample:
    :param idx:  单条idx 是应该有顺序的
    :param star:
    :return:

    '''
    ele_token = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]#[i for i in range(4,56)]
    sep = [3]
    bracket_pre = [30]#[70]
    bracket_post = [31]#[71]

    res = 0

    if idx==0:
        return res,star
    if sample[idx] not in ele_token:

            j = idx
    else:
        j = idx-1


    while j>=0:

        if sample[j] in  sep and j!=0 :
            '''
            跨范围的一般不是特殊符号
            '''
            temp,idx_star = find_in_other_frag(sample,j,star) # 第i 条 star 记录
            # print(temp)
            res = deepcopy(temp)
            star[idx_star]+=1
            break

        if sample[j] in ele_token:
            res = j
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


    return res,star


def find_root_smi(pred_codes):

        star = np.where((np.array(pred_codes)==34) | (np.array(pred_codes)==35),1,0)
        sample = pred_codes
        ele_token = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        root_idxs = np.zeros(len(pred_codes))
        root_root_idxs = np.zeros((len(pred_codes)))
        root_root_root_idxs = np.zeros((len(pred_codes)))
        for idx in range(len(pred_codes)):
            # idx = 15
            root_idx , new_star_single = find_root_smi_cur(sample,idx,copy.deepcopy(star))
            root_idx = root_idx
            root_idxs[idx] = root_idx
            if sample[idx] in ele_token:
                root_root_idx = root_idxs[root_idx]
                root_root_idxs[idx]   = int(root_root_idx)
                root_root_root_idx = root_idxs[int(root_root_idx)]
                root_root_root_idxs[idx] = int(root_root_root_idx)
                star = new_star_single
            else:
                root_root_idxs[idx] = root_idx
                root_root_root_idxs[idx] = root_idx
            # print(idx,sample[idx],root_idx,new_star_single)
            # break
        root_idxs = root_idxs.astype('int')
        root_root_idxs = root_root_idxs.astype('int')
        root_root_root_idxs = np.rint(root_root_root_idxs).astype('int')

        return root_idxs.tolist(),root_root_idxs.tolist(),root_root_root_idxs.tolist() #,star.tolist()


if __name__=='__main__':
    mol = Chem.MolFromPDBFile('/home/bairong/data/800w/800w_druglike/5119647.pdb')


    fragUtil = FragmolUtil()

    res, positions, neighbor, orig_ind, frags = fragUtil.flatten_seq(mol)
    print(res)
    sstring = ''.join([ss for ss in res])
    sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")
    sstring = sstring.replace("@@", "V")
    sstring = sstring.replace("([*])", "L")
    sstring = sstring.replace("[*]", "M")
    print(sstring)
    code = fragUtil.encode(res)
    root,root1,root2 = find_root_smi(code)

    # res = []
    # for s in sstring:
    #     if s=='X':
    #         res.append('Cl')
    #         continue
    #     if s=='Y':
    #         res.append('[nH]')
    #         continue
    #     if s=='Z':
    #         res.append('Br')
    #         continue
    #     if s=='V':
    #         res.append('@@')
    #         continue
    #     if s=='L':
    #         res.append('([*])')
    #         continue
    #     if s=='M':
    #         res.append('[*]')
    #         continue
    #     else:
    #         res.append(s)
    for i in range(len(code)):

        print('{:10} {:10} {:10} {:10} {:10} {:10} '.format(str(i), str(code[i]),str(fragUtil.vocab_i2c_v1_decode[code[i]]),str(int(root[i])),str(int(root1[i])),str(int(root2[i]))))






