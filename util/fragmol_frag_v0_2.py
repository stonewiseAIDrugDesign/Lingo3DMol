# This file is part of Lingo3DMol
#
# Lingo3DMol is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Lingo3DMol is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Lingo3DMol. If not, see <https://www.gnu.org/licenses/>.



import os
os.sys.path.append(os.pardir)
import rdkit
from rdkit import Chem
from copy import deepcopy
import numpy as np
from rdkit.Chem import AllChem, rdGeometry
from util.find_root import find_root_smi
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import traceback
from util.rotb import rotatable_bond
class FragmolUtil():
    def __init__(self): #TODO  code 还可以改

        self.vocab_list = ["pad", "start", "end", 'sep',
                           "C", "c", "N", "n", "S", "s", "O", "o",
                           "F",
                           "X", "Y", "Z",
                           "/", "\\", "@", "V", "H",  # W for @， V for @@
                           # "Cl", "[nH]", "Br", # "X", "Y", "Z", for origin
                           "1", "2", "3", "4", "5", "6",
                           "#", "=", "-", "(", ")", "[", "]", "M" ,"L" # Misc
                           ]

        self.vocab_i2c_v1 = {i: x for i, x in enumerate(self.vocab_list)}

        self.vocab_c2i_v1 = {self.vocab_i2c_v1[i]: i for i in self.vocab_i2c_v1}

        self.vocab_list_decode = ["pad", "start", "end", "sep",
                                  "C", "c", "N", "n", "S", "s", "O", "o",
                                  "F",
                                  "Cl", "[nH]", "Br",
                                  "/", "\\", "@", "@@", "H",  # W for @， V for @@
                                  # "Cl", "[nH]", "Br", # "X", "Y", "Z", for origin
                                  "1", "2", "3", "4", "5", "6",
                                  "#", "=", "-", "(", ")", "[", "]", "[*]","([*])"  # Misc
                                  ]

        self.vocab_i2c_v1_decode = {i: x for i, x in enumerate(self.vocab_list_decode)}

        self.vocab_c2i_v1_decode = {self.vocab_i2c_v1_decode[i]: i for i in self.vocab_i2c_v1_decode}

        self.encode_length = 100

        self.ele_token = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        self.resolution = 0.1

        self.dist_grid = {0: 0, 10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6, 16: 7, 17: 8, 18: 9, 19: 10, 20: 11, 21: 12}

        self.vocab_list_decode_new = [
            "pad_0", "start_0", "end_0", 'sep_0',
            "C_0", "C_5", "C_6", "C_10", "C_11", "C_12",
            "c_0", "c_5", "c_6", "c_10", "c_11", "c_12",
            "N_0", "N_5", "N_6", "N_10", "N_11", "N_12",
            "n_0", "n_5", "n_6", "n_10", "n_11", "n_12",
            "S_0",
            "s_0", "s_5", "s_6", "s_10", "s_11", "s_12",
            "O_0", "O_5", "O_6", "O_10", "O_11", "O_12",
            "o_0", "o_5", "o_6", "o_10", "o_11", "o_12",
            "F_0",
            "Cl_0",
            "[nH]_0", "[nH]_5", "[nH]_6", "[nH]_10", "[nH]_11", "[nH]_12",
            "Br_0",
            "/_0", "\\_0", "@_0", "@@_0", "H_0",
            "1_0", "2_0", "3_0", "4_0", "5_0", "6_0",
            "#_0", "=_0", "-_0", "(_0", ")_0", "[_0", "]_0", "[*]_0","([*])_0"
        ]




        self.vocab_i2c_v1_decode_new = {i: x for i, x in enumerate(self.vocab_list_decode_new)}

        self.vocab_c2i_v1_decode_new = {self.vocab_i2c_v1_decode_new[i]: i for i in self.vocab_i2c_v1_decode_new}

    def mol_with_atom_index(self, mol):

        for atom in mol.GetAtoms():
            atom.SetProp('atomNum', str(atom.GetIdx()))

        return mol

    def tree_decomp(self, mol):

        for i in mol.GetAtoms():
            i.SetIntProp("atom_idx", i.GetIdx())

        for i in mol.GetBonds():
            i.SetIntProp("bond_idx", i.GetIdx())

        cut_bonds = []

        for bond in mol.GetBonds():
            bond_type = bond.GetBondTypeAsDouble()
            bond_idx = bond.GetIdx()

            if bond.IsInRing():
                continue
            if bond_type != 1.0:
                continue

            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.IsInRing() or a2.IsInRing():
                if a1.GetSymbol()=='H' or a2.GetSymbol()=='H':
                    continue
                cut_bonds.append(bond_idx)
                # print(a1.GetSymbol(),a2.GetSymbol())

        frags = Chem.FragmentOnBonds(mol, cut_bonds)
        frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=True)

        cliques = []
        for frag in frags:
            frag = Chem.RWMol(frag)
            temp = []
            for atom in frag.GetAtoms():
                if atom.GetSymbol() == '*':
                    continue
                else:
                    temp.append(int(atom.GetProp('atom_idx')))
            cliques.append(deepcopy(temp))
        return cliques, frags

    def get_clique_mol(self, final_frag, index, root_index, already=[], ind=None, pos_linker=None, removeFirst=True):

        '''
        :param final_frag: 目前要处理的片段 mol
        :param index:      该片段上的连接位点
        :param already:    该片段上的idx
        :param ind:        总的来说 第几个片段
        :param removeFirst:
        :return:
        '''

        candidates = already
        candidates_ind = ind

        c = already

        # remove rooted * if exist (not first frag)
        if removeFirst:
            flag = False
            for i in range(final_frag.GetNumAtoms()):
                atom = final_frag.GetAtomWithIdx(i)
                idx = int(atom.GetProp('atomNum'))
                if atom.GetSymbol() != '*' and idx == index:  # 找到切割的原子
                    neighbors = atom.GetNeighbors()
                    delete = []
                    for n in neighbors:
                        if n.GetSymbol() == '*':
                            p = final_frag.GetConformer().GetAtomPosition(n.GetIdx())
                            if p.x == pos_linker[0] and p.y == pos_linker[1] and p.z == pos_linker[2]:
                                delete.append(n.GetIdx())
                                flag = True
                                break

                    delete = sorted(delete, reverse=True)  # TODO
                    for d in delete:
                        mw = Chem.RWMol(final_frag)

                        mw.RemoveAtom(d)
                        final_frag = mw.GetMol()
                    if flag:
                        break
        
        # rootedAt rooted_idx

        rooted_idx = -1
        for i in range(final_frag.GetNumAtoms()):
            atom = final_frag.GetAtomWithIdx(i)
            if atom.GetSymbol() != '*':
                idx = int(atom.GetProp('atomNum'))
                if idx == index:
                    rooted_idx = atom.GetIdx()
                    break

        smi = Chem.MolToSmiles(final_frag, rootedAtAtom=rooted_idx)
        r_order1 = final_frag.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
        rorder_model1 = Chem.RenumberAtoms(final_frag, r_order1)

        # final smiles
        smi = Chem.MolToSmiles(rorder_model1, rootedAtAtom=0)
        final_smi, next_nhr = self.parse_star(smi)

        # next neighbor
        next_ = []
        #     print('--------------')
        for i in range(rorder_model1.GetNumAtoms()):
            atom = rorder_model1.GetAtomWithIdx(i)
            s = atom.GetSymbol()
            if s == '*':
                neighbor = atom.GetNeighbors()
                for n in neighbor:
                    idx = n.GetIdx()
                    orig_idx = int(n.GetProp('atomNum'))
                    if orig_idx in c:
                        next_.append(orig_idx)

        final_next = [[a1, a2] for a1, a2 in zip(next_, next_nhr)]

        position = []

        for i in range(0, rorder_model1.GetNumAtoms()):
            a = rorder_model1.GetAtomWithIdx(i)
            s = a.GetSymbol()
            if s != '*':
                pos = rorder_model1.GetConformer().GetAtomPosition(i)
                position.append([pos.x, pos.y, pos.z])
            
        neighbor = []
        orig_idx = []

        for i in range(0, rorder_model1.GetNumAtoms()):
            atom = rorder_model1.GetAtomWithIdx(i)
            nei = atom.GetNeighbors()
            cur_idx = atom.GetIdx()
            idx = []
            s = atom.GetSymbol()
            if s == '*': continue
            orig_idx.append(int(atom.GetProp('atomNum')))

            if cur_idx == 0:
                neighbor.append(root_index)
                continue
            for n in nei:
                n_idx = n.GetIdx()
                if n_idx < cur_idx:
                    idx.append(int(n.GetProp('atomNum')))

            new_idx = sorted(idx)

            neighbor.append(new_idx[-1])
        # final smi:返回 连接位点 为 0 的frag
        # final next: root idx 和 star 对应的 global idx
        # 第几个片段
        # 重原子 对应的 坐标
        # 邻居
        # 当前原子的 idx
        # 当前的片段 reorder 根据 smiles 顺序
        return final_smi, final_next, candidates_ind, position, neighbor, orig_idx, rorder_model1

    def parse_star(self, smi):
        label = []

        ss = smi[::-1]
        for i, s in enumerate(ss):
            if s == '*':
                for j in range(i, len(ss)):
                    if ss[j] == '[':
                        label.append([len(ss) - j - 1, len(ss) - i - 1])
                        break
        res = ''
        next_ = []
        if len(label) > 0:
            for pre, post in label:
                res = smi[:pre + 1] + smi[post:]
                next_.append(int(smi[pre + 1:post]))
                smi = res

            return res, next_[::-1]
        else:
            return smi, []

    def flatten_seq(self, mol, initial_index=0):
        mol = self.mol_with_atom_index(mol)
        cliques, frags = self.tree_decomp(mol)
        # print(cliques)
        res = []
        already = []
        pre_v = []
        next_ = []
        flag = False
        positions = []
        neighbors = []
        orig_indices = []
        final_frags = []

        for idx, (c, frag) in enumerate(zip(cliques, frags)):

            if initial_index in c:

                smi, new_next_, can_ind, coords, neighbor, orig_ind, frag_ = self.get_clique_mol(frag, initial_index,
                                                                                                 initial_index, c, idx,
                                                                                                 removeFirst=flag)

                flag = True
                res.append(smi)
                already.append(can_ind)
                positions.extend(coords)
                neighbors.extend(neighbor)
                orig_indices.extend(orig_ind)
                final_frags.append(frag_)

                for n in new_next_:
                    if sorted(n) not in pre_v and sorted(n) not in next_:

                        next_=[n]+next_

        while len(next_) > 0:
            linker_atom, nhr_atom = next_.pop(0)
            pre_v.append(sorted([linker_atom, nhr_atom]))
            for idx, (c, frag) in enumerate(zip(cliques, frags)):

                if idx in already:
                    continue
                if nhr_atom in c:
                    p = mol.GetConformer().GetAtomPosition(linker_atom)
                    pos = [p.x, p.y, p.z]
                    smi, new_next_, can_ind, coords, neighbor, orig_ind, frag_ = self.get_clique_mol(frag, nhr_atom,
                                                                                                     linker_atom, c,
                                                                                                     idx, pos)

                    if can_ind not in already:
                        res.append(smi)
                        already.append(can_ind)
                        positions.extend(coords)
                        neighbors.extend(neighbor)
                        orig_indices.extend(orig_ind)
                        final_frags.append(frag_)
                        for n in new_next_:
                            if sorted(n) not in pre_v and sorted(n) not in next_:
                                next_ = [n] + next_

        return res, positions, neighbors, orig_indices, final_frags

    def idxSMIMap(self, smi_map,smi_map_n1,smi_map_n2, position):


        nei_position = np.zeros((len(smi_map), 3))
        nei_position_n1 = np.zeros((len(smi_map), 3))
        nei_position_n2 = np.zeros((len(smi_map), 3))

        for i in range(len(smi_map)):
            nei_position[i] = position[smi_map[i]]
            nei_position_n1[i] = position[smi_map_n1[i]]
            nei_position_n2[i] = position[smi_map_n2[i]]

        return  nei_position, nei_position_n1, nei_position_n2

    def encode(self, res):
        final = [1]
        try:
            notfinish_flag = False
            for i,sstring in enumerate(res):
                sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")
                sstring = sstring.replace("@@", "V")
                sstring = sstring.replace("([*])","L")
                sstring = sstring.replace("[*]","M")

                smi = []
                flag = False
                for xchar in sstring:

                    if xchar in self.vocab_c2i_v1.keys():
                        smi.append(self.vocab_c2i_v1[xchar])
                    else:
                        flag = True
                        break
                if flag:
                    notfinish_flag = True
                    break
                smi +=[3]
                final.extend(smi)

            if not notfinish_flag:
                final.extend([2,3])

            smile = final + (self.encode_length - len(final)) * [0]
            if sum(smile[1:])==0:
                raise 'invalid encode mol'
            return smile[:self.encode_length]
        except Exception as e:
            print(f'there is an exception {e}')
            return None

    def generate_coords(self, sstring, coords,root):

        new_coords = np.zeros((len(sstring), 3))

        c_idx = 0

        for i in range(len(sstring)):
            s = sstring[i]
            if s in self.ele_token or s==-1:
                new_coords[i] = coords[c_idx]
                c_idx += 1
            else:

                root_idx = root[i]
                new_coords[i]  = new_coords[root_idx]

        return new_coords

    def rotate(self, coords, rotMat, center=(0, 0, 0)):

        """
        Rotate a selection of atoms by a given rotation around a center
        """

        newcoords = coords - center

        return np.dot(newcoords, np.transpose(rotMat)) + center

    def position_degree(self, neighbor, neighbor_n1, pos):

        thetas = np.zeros(len(pos))
        for i in range(len(pos)):
            orig = neighbor[i]
            pre = neighbor_n1[i]
            cur = pos[i]
            if np.sum(cur) == 0 or np.sum(pre) == 0 or np.sum(orig) == 0: 
                continue
            
            cur_vec = (cur - orig) / np.linalg.norm((cur - orig))
            pre_vec = (pre - orig) / np.linalg.norm((pre - orig))
            cos_theta = np.arccos(np.sum(cur_vec * pre_vec))
            theta = np.degrees(cos_theta)
            thetas[i] = theta

        return thetas

    def isInplane(self, orig, pre, cur):

        cur_vec = (cur - orig) / np.linalg.norm((cur - orig))
        pre_vec = (pre - orig) / np.linalg.norm((pre - orig))
        cos_theta = np.arccos(np.sum(cur_vec * pre_vec))
        theta = np.degrees(cos_theta)
        if abs(theta - 180) == 0:
            return False
        return True

    def cross(self, orig, pre, third):
        p1 = pre - orig
        p2 = third - orig
        vec1 = p1 / np.linalg.norm(p1)
        vec2 = p2 / np.linalg.norm(p2)

        '''
        叉乘: 右手法则，
        '''

        normal_vector = np.cross(vec1, vec2)

        return normal_vector

    def plane_vec(self, pre_v1, pre_v2):

        '''
        :param pre_v1:
        :param pre_v2:
        :return:
        '''

        p_v1 = pre_v1 / np.linalg.norm(pre_v1)
        p_v2 = pre_v2 / np.linalg.norm(pre_v2)

        cos_theta = np.arccos(np.sum(p_v1 * p_v2))
        theta = np.degrees(cos_theta)

        return theta

    def plane_degree(self, neighbor, neighbor_n1, neighbor_n2, pos):

        degree = np.zeros(len(pos))

        for i in range(len(pos)):
            orig = neighbor[i]
            pre = neighbor_n1[i]
            pre_pre = neighbor_n2[i]
            cur = pos[i]
            positions0 = [orig, pre, pre_pre]
            positions1 = [orig, pre, cur]

            if np.sum(cur) == 0.0 or np.sum(pre) == 0.0 or np.sum(orig) == 0.0 or np.sum(
                    pre_pre) == 0.0:  # 无意义，第二个，第一个 无法成角度的点,避免/0
                # print('--------------------',np.sum(cur),np.sum(pre),np.sum(orig),np.sum(pre_pre))
                continue

            '''
            exclude points that cannot be in the same plane
            '''

            isp = self.isInplane(orig, pre, pre_pre)

            if isp is False:
                continue

            isp = self.isInplane(orig, pre, cur)

            if isp is False:
                continue

            pre_v1 = self.cross(orig, pre, pre_pre)
            pre_v2 = self.cross(orig, pre, cur)
            try:
                cos_theta = np.arccos(np.sum(pre_v1 / np.linalg.norm(pre_v1) * pre_v2 / np.linalg.norm(pre_v2)))
            except:
                pass
            theta = np.degrees(cos_theta)
            degree[i] = theta

        return degree

    def RingLarge(self,mol):
        '''
        :param mol: obj
        :return:    0: no ring 1: single ring 2: double ring 3: triple ring
        '''
        ssr = [[list(x), 1] for x in Chem.GetSymmSSSR(mol)]
        for i in range(len(ssr)):

            if len(ssr[i][0]) >= 7:
                
                return True
            if len(ssr[i][0])<=4:
                return True
        return False

    def position_degree1(self,neighbor, neighbor_n1, pos):

        orig = neighbor
        pre = neighbor_n1
        cur = pos
        cur_vec = (cur - orig) / np.linalg.norm((cur - orig))
        pre_vec = (pre - orig) / np.linalg.norm((pre - orig))
        cos_theta = np.arccos(np.sum(cur_vec * pre_vec))
        theta = np.degrees(cos_theta)

        return theta


    def acute_angle(self,mol):
        ssr = [list(r) for r in Chem.GetSymmSSSR(mol)]
        conf = mol.GetConformer()
        for ring in ssr:
            if len(ring) <= 4:
                # print(list(ring))
                continue
            
            for idx in ring:

                atom = mol.GetAtomWithIdx(idx)
                neighbors = atom.GetNeighbors()
                neighbor = []
                for n in neighbors:

                    if n.IsInRing() and n.GetIdx() in list(ring):
                        neighbor.append(n)
                    if len(neighbor) == 2:
                        break

                p = conf.GetAtomPosition(idx)
                orig = np.array([p.x, p.y, p.z])
                p = conf.GetAtomPosition(neighbor[0].GetIdx())
                n1 = np.array([p.x, p.y, p.z])
                p = conf.GetAtomPosition(neighbor[1].GetIdx())
                n2 = np.array([p.x, p.y, p.z])
                theta = self.position_degree1(orig, n1, n2)
                if theta < 80 or theta>160:
                    return False

        return True

    def valid_patt(self,mol):

        # hasalert = alertfrag(mol)
        #
        # if hasalert:
        #     return False

        ssr = [[list(x), 1] for x in Chem.GetSymmSSSR(mol)]

        if len(ssr)==0:
            patt_cdouble = Chem.MolFromSmarts("C=C")
            atomids = mol.GetSubstructMatches(patt_cdouble)
            if len(atomids)>=2:
                return False


        return True

    def valid_RingCluster(self,mol,cluster=3):

        '''
        :param mol: obj
        :return:    0: no ring 1: single ring 2: double ring 3: triple ring
        '''

        ssr = [[list(x), 1] for x in Chem.GetSymmSSSR(mol)]
        # Merge Rings with intersection > 2 atoms
        for i in range(len(ssr)):
            for j in range(i + 1, len(ssr)):
                inter = set(ssr[i][0]) & set(ssr[j][0])
                if len(inter)==1:
                    return False
                if len(inter)>2:
                    return False

                if len(inter) >= 2:
                    ssr[i][0].extend(ssr[j][0])
                    ssr[i][0] = sorted(set(ssr[i][0]), key=ssr[i][0].index)
                    ssr[i][1] = ssr[i][1] + 1
                    ssr[j][0] = []
                    ssr[j][1] = 0
        cluster_num = [n[1] for n in ssr]

        if len(cluster_num)==0:
            return True

        if max(cluster_num)>=3:
            return False

        return True

    def get_atom_type_frag(self, mol,res):

        try:
            smi = Chem.MolToSmiles(mol)


            m1 = Chem.MolFromSmiles(smi)
            rotb = rotatable_bond(m1)

        except:
            atom_type = [-1] * mol.GetNumAtoms()
            # 不要 * 号
            for i in range(mol.GetNumAtoms() - 1, -1, -1):
                a = mol.GetAtomWithIdx(i)
                s = a.GetSymbol()
                if s == '*':
                    atom_type.pop(i)
            return deepcopy(atom_type)

        if rotb>=3:
            atom_type = [-2] * mol.GetNumAtoms()
            # 不要 * 号
            for i in range(mol.GetNumAtoms() - 1, -1, -1):
                a = mol.GetAtomWithIdx(i)
                s = a.GetSymbol()
                if s == '*':
                    atom_type.pop(i)
            return deepcopy(atom_type)

        # check alert
        noalert = self.valid_patt(mol)

        #check large ring
        large_ring = self.RingLarge(mol)

        # check unnormal conformer
        valid_conf = self.acute_angle(mol)

        if  large_ring or not noalert or not valid_conf:
            atom_type = [-1] * mol.GetNumAtoms()

            # 不要 * 号
            for i in range(mol.GetNumAtoms() - 1, -1, -1):
                a = mol.GetAtomWithIdx(i)
                s = a.GetSymbol()
                if s == '*':
                    atom_type.pop(i)

            return deepcopy(atom_type)


        # check large,spiral,triple ring
        validRing = self.valid_RingCluster(mol)
        if not validRing:
            atom_type = [-1] * mol.GetNumAtoms()
            # 不要 * 号
            for i in range(mol.GetNumAtoms() - 1, -1, -1):
                a = mol.GetAtomWithIdx(i)
                s = a.GetSymbol()
                if s == '*':
                    atom_type.pop(i)
            return deepcopy(atom_type)

        ssr = [list(r) for r in Chem.GetSymmSSSR(mol)]



        # inter pair
        atom_type = [0] * mol.GetNumAtoms()
        flag = False
        for i in range(len(ssr)):
            for j in range(i + 1, len(ssr)):
                inter = set(ssr[i]) & set(ssr[j])
                union_type = str(len(ssr[i]) + len(ssr[j]))
                if len(inter) > 0 and len(inter) != 2:
                   flag=True
                for inter_idx in inter:
                    atom_type[inter_idx] = union_type

        if flag:
            atom_type = [-1] * mol.GetNumAtoms()
            # 不要 * 号
            for i in range(mol.GetNumAtoms() - 1, -1, -1):
                a = mol.GetAtomWithIdx(i)
                s = a.GetSymbol()
                if s == '*':
                    atom_type.pop(i)
            return deepcopy(atom_type)

        # the other ring atom
        for i in range(len(ssr)):
            ring = ssr[i]
            for r in ring:
                if atom_type[r] == 0:
                    atom_type[r] = len(ring)

        # 不要 * 号
        for i in range(mol.GetNumAtoms() - 1, -1, -1):
            a = mol.GetAtomWithIdx(i)
            s = a.GetSymbol()
            if s == '*':
                atom_type.pop(i)

        return deepcopy(atom_type)


    def atom_type(self, sstring, frags,reses):

        result = []
        for i,mol in enumerate(frags):
            smi = reses[i]
            atom_type = self.get_atom_type_frag(mol,reses[i])  # 按照 index 顺序判断的, unknown 是 -1
            result.extend(deepcopy(atom_type))

        c_idx = 0
        new_result = np.zeros(len(sstring))
        for i in range(len(sstring)):
            s = sstring[i]
            if s in self.ele_token:
                new_result[i] = result[c_idx]
                c_idx += 1

        return new_result

    def recode(self, sstring, atom_type):
        new_sstring = []
        token = ''
        last_sep = 0
        flag = False
        for i, (s, t) in enumerate(zip(sstring, atom_type)):
            if t==-1:
                break
            if t==-2:
                break

            d_s = self.vocab_i2c_v1_decode[s]
            if d_s=='sep':
                last_sep = i
            new_token = d_s + '_' + str(int(t))
            if new_token not in self.vocab_c2i_v1_decode_new.keys():
                flag=True
                break
            new_sstring.append(self.vocab_c2i_v1_decode_new[new_token])
            token += new_token

        if flag:
            new_sstring = new_sstring[:last_sep]
        new_sstring1 = new_sstring+(self.encode_length - len(new_sstring)) * [0]
        s = sum(new_sstring1[1:])
        if s==0:
            raise# 'invalid recode'
        return new_sstring1

    def recode_coords(self,new_code,new_position):

        idx = len(new_code)

        for i in range(len(new_position)):
            pos = new_position[i]

            if np.min(np.array(pos)) < 0 or np.max(np.array(pos)) >= 240:
                idx = i
                break

        for i in range(idx,len(new_code)):
            new_code[i]=0
            new_position[i][0] = 0
            new_position[i][1] = 0
            new_position[i][2] = 0


        return new_code,new_position

    def encode_mol(self, mol, center, rrot=None):

        resolution = self.resolution
        length = int(24 / resolution)
        grid_c = (length - 1) / 2

        res, positions, neighbor, orig_ind, frags = self.flatten_seq(mol)
        if rrot is not None:
            positions = self.rotate(positions, rrot, center=center)
        new_pdb_coords1 = (np.array(positions) - center) / resolution + np.array([grid_c, grid_c, grid_c])
        ll1 = np.rint(new_pdb_coords1).astype('int')
        code = self.encode(res)
        atom_type = self.atom_type(code, frags, res)
        new_code = self.recode(code, atom_type)

        smi_map,smi_map_n1,smi_map_n2 = find_root_smi(code)
        new_position = self.generate_coords(code, ll1,smi_map)
        neighbor_positoins, neighbor_positoins_n1, neighbor_positoins_n2 = self.idxSMIMap(smi_map,smi_map_n1,smi_map_n2, new_position)

        dist = np.rint(np.clip(np.sqrt(np.sum(np.square(np.array(new_position) - np.array(neighbor_positoins)), axis=-1)), a_min=0, a_max=21)).astype('int')
        new_dist = []
        for d in dist:
            new_dist.append(self.dist_grid[d])
        theta = self.position_degree(neighbor_positoins, neighbor_positoins_n1, new_position)
        theta = np.clip(np.rint(theta).astype('int'), a_max=180, a_min=0)
        degree = self.plane_degree(neighbor_positoins, neighbor_positoins_n1, neighbor_positoins_n2, new_position)
        degree = np.clip(np.rint(degree).astype('int'), a_min=0, a_max=180)

        new_code1,new_position1 = self.recode_coords(new_code,new_position)

        return new_code1, new_position1, smi_map, smi_map_n1, smi_map_n2, theta, new_dist, degree  # ,smi_map,smi_map_n1,smi_map_n2,

    def mergeSmi(self, rwmol, smi, uuid):

        try:

            mol = Chem.MolFromSmiles(smi, sanitize=False)
            # label conn atom
            nbr_idx = -1
            orig_idx = -1
            orig_symbol = None

            for i in range(rwmol.GetNumAtoms()-1,-1,-1):
                atom = rwmol.GetAtomWithIdx(i)
                s = atom.GetSymbol()
                if s == '*':
                    nbr_idx = atom.GetIdx()
                    n = atom.GetNeighbors()
                    nbr = n[0]
                    orig_idx = nbr.GetIdx()
                    orig_symbol = nbr.GetSymbol()
                    atom1 = rwmol.GetAtomWithIdx(orig_idx)
                    atom1.SetProp('delete', str(uuid))
                    break

            if nbr_idx == -1:
                return rwmol, True  # return modified mol, isFinish true

            # remove fake atom
            rwmol.RemoveAtom(nbr_idx)
            node_to_idx = {}
            # add mol
            #
            pre = rwmol.GetNumAtoms()
            if mol is None:
                return None,True
            combo = Chem.CombineMols(rwmol, mol)

            # connect two atom
            # 1 find pre orig atom
            pre_index = -1
            for i in range(rwmol.GetNumAtoms()):
                atom = rwmol.GetAtomWithIdx(i)
                labels = atom.GetPropsAsDict()
                if 'delete' not in labels.keys():
                    continue
                label = labels['delete']
                if str(label) == str(uuid):
                    pre_index = atom.GetIdx()
                    break

            # 2 find cur conn atom (fake)
            # cur_idx = node_to_idx[1 + pre]

            # 3 connect
            ecombo = Chem.RWMol(combo)
            ecombo.AddBond(pre_index, pre, Chem.BondType.SINGLE)

            return ecombo, False
        except Exception as e:
            print(f'{e}')
            return None, True

    def mergeSmiles3D(self, res, position):
        try:

            m = rdkit.Chem.RWMol()
            node_to_idx = {}
            conn = []

            mol = Chem.MolFromSmiles(res[0])


            m = Chem.RWMol(mol)
            for i, r in enumerate(res):
                if i == 0:

                    continue
                m, isFinish = self.mergeSmi(m, r, i)
                if m is None:
                    print('invalid m is None 2')
                    return None, None
                if isFinish:
                    break


            # delete unnecessary fake atom
            fakes = []
            for i in range(m.GetNumAtoms()):
                atom = m.GetAtomWithIdx(i)
                s = atom.GetSymbol()
                if s == '*':
                    fakes.append(atom.GetIdx())
            if len(fakes)>0:
                print('exist fake atom')
                return None,None
            fakes = sorted(fakes, reverse=True)
            for f in fakes:
                # m.ReplaceAtom(f, Chem.Atom('H'))
                m.RemoveAtom(f)
            Chem.SanitizeMol(m)
            m = Chem.RemoveHs(m)

            print('finish topo')
            if m.GetNumAtoms() != len(position):
                print(len(position), m.GetNumAtoms())
                return None, None

            conf = Chem.Conformer(m.GetNumAtoms())

            # conf = m.GetConformer()

            for i in range(m.GetNumAtoms()):
                pos = position[i]
                p = rdGeometry.Point3D(pos[0], pos[1], pos[2])
                conf.SetAtomPosition(i, p)
            m.AddConformer(conf)
            m.SetProp('_Name', Chem.MolToSmiles(m))
            Chem.SanitizeMol(m)

            # print('geo')
            if m is None:
                return None, None

            return Chem.MolToSmiles(m), m

        except Exception as e:
            print(f'invalid {e} {traceback.format_exc()}')

            return None,None

    def decode3d(self, batch_codes, positions):

        gen_smiles = []

        tokens = []

        pos_res = []

        positions = positions.astype(np.float16)

        positions = positions.tolist()

        for i, sample in enumerate(batch_codes):
            try:

                res = []
                csmile = ""
                token = []
                pos = positions[i]
                pos_temp = []

                for j, xchar in enumerate(sample[:]):
                    t = self.vocab_i2c_v1_decode_new[xchar]
                    new_t = t.split('_')[0]

                    if xchar == 2:
                        res.append(deepcopy(csmile))
                        csmile = ""
                        break

                    if xchar == 1:
                        continue

                    token.append(t)
                    if xchar == 3:
                        res.append(deepcopy(csmile))
                        csmile = ""
                        continue

                    if xchar == 1:
                        continue

                    csmile += new_t
                    if self.vocab_c2i_v1_decode[new_t] in self.ele_token:
                        pos_temp.append(pos[j])
                tokens.append(deepcopy(token))
                gen_smiles.append(deepcopy(res))
                pos_res.append(deepcopy(pos_temp))
            except Exception as e:
                print(f'decode {e}')
                continue
        reses = []
        moleculars = []
        print(len(gen_smiles))
        for i, res in enumerate(gen_smiles):

            try:
                if len(res) > 0:
                    smi, m = self.mergeSmiles3D(res[:-1], pos_res[i])
                    reses.append(smi)
                    moleculars.append(m)

                else:

                    reses.append(None)

                    moleculars.append(None)

            except:

                reses.append(None)

                moleculars.append(None)

        return reses, tokens, moleculars


if __name__ == '__main__':

    fragUtil = FragmolUtil()

    mol = Chem.SDMolSupplier('/home/bairong/data/pocket_data1/ligand/1akt_ligand.sdf')[0]
    print(Chem.MolToSmiles(mol))


    res = fragUtil.flatten_seq(mol)

    print(res)

    code,  new_position, neighbor_positoins, neighbor_positoins_n1, neighbor_positoins_n2, theta, new_dist, degree = fragUtil.encode_mol(
        mol, center=np.array([0, 0, 0]))
    for i, r in enumerate(zip(code, new_position, neighbor_positoins, neighbor_positoins_n1,neighbor_positoins_n2, theta, new_dist, degree)):

        print(fragUtil.vocab_i2c_v1_decode_new[code[i]], i, r,new_dist[i],theta[i],degree[i])
