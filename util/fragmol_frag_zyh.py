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
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import traceback


class FragmolUtil():
    def __init__(self): #TODO  code 还可以改

        self.vocab_list = ["pad", "start", "end", 'sep',
                           "C", "c", "N", "n", "S", "s", "O", "o",
                           "F",
                           "X", "Y", "Z",
                           "/", "\\", "@", "V", "H",  # W for @， V for @@
                           # "Cl", "[nH]", "Br", # "X", "Y", "Z", for origin
                           "1", "2", "3", "4", "5", "6",
                           "#", "=", "-", "(", ")", "[", "]", "M" ,"L","+" # Misc
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
                                  "#", "=", "-", "(", ")", "[", "]", "[*]","([*])","+"  # Misc
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
            "o_0", "o_5", "o_6", "+_0", "o_11", "o_12",
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
                #         print(labels.items())
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
                # print(Chem.MolToSmiles(m,rootedAtAtom=0))
                if m is None:
                    print('invalid m is None 2')
                    return None, None
                # print(r,Chem.MolToSmiles(m))
                if isFinish:
                    break

            # delete unnecessary fake atom
            # fakes = []
            # for i in range(m.GetNumAtoms()):
            #     atom = m.GetAtomWithIdx(i)
            #     s = atom.GetSymbol()
            #     if s == '*':
            #         fakes.append(atom.GetIdx())

            # if len(fakes)>0:
            #     print('exist fake atom')
            #     return None,None
            # fakes = sorted(fakes, reverse=True)
            # for f in fakes:
            #     # m.ReplaceAtom(f, Chem.Atom('H'))
            #     m.RemoveAtom(f)
            # Chem.SanitizeMol(m)
            # m = Chem.RemoveHs(m)
            # import pdb;pdb.set_trace()

            print('finish topo')
            # if m.GetNumAtoms() != len(position):
            #     print(len(position), m.GetNumAtoms())
            #     return None, None

            conf = Chem.Conformer(m.GetNumAtoms())

            # conf = m.GetConformer()
            pos_num=0
            for i in range(m.GetNumAtoms()):
                atom_symbol = m.GetAtomWithIdx(i).GetSymbol()
                if atom_symbol=='*':
                    # atom_symbol='H'
                    continue
                pos = position[pos_num]
                p = rdGeometry.Point3D(pos[0], pos[1], pos[2])
                conf.SetAtomPosition(i, p)
                pos_num+=1

            m.AddConformer(conf)
            m.SetProp('_Name', Chem.MolToSmiles(m))
            Chem.SanitizeMol(m)
            # import pdb;pdb.set_trace()

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

                    if xchar == 2 or xchar== 0:
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
        # import pdb;pdb.set_trace()
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

