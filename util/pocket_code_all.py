# This file is part of Lingo3DMol
#
# Lingo3DMol is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Lingo3DMol is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Lingo3DMol. If not, see <https://www.gnu.org/licenses/>.



import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import RDConfig
import os
from scipy.spatial.distance import cdist

class PocketCode():
    def __init__(self):
        self.symbol = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Se', 'P','Ca','Zn','Centroid']  # [ 'C', 'O', 'N', 'F','Cl', 'Se', 'P', 'S', 'Ca']
        self.res = ['0O6', 'GHP', 'HG7', 'ALA', 'ASP', 'TRP', 'M3L', 'MP4',
                    'PHD', 'ARG', 'DVA', '3PL', 'FH7', 'PRO', 'GBL', 'MSE',
                    'FTR', '4AK', 'YCP', 'X95', 'UNK', 'CA', 'NEP', 'SER',
                    'MCM', 'CMT', 'SLL', 'PLM', 'FDL', 'LTN', 'ILE', 'FGA',
                    'NHS', 'HPH', 'ALC', 'MK8', 'MDF', '6CW', 'SRZ', 'OMX',
                    'MTY', 'K26', '4FO', 'DAR', 'DIL', '4H0', 'OMZ', 'DAL',
                    'KPI', 'CPI', 'HCL', 'ALY', 'HIC', 'DPN', 'C8V', 'DGL',
                    'TYS', 'MLY', 'GLN', 'GLY', 'ALQ', 'DTH', 'TYJ', 'SCY',
                    'FPK', 'MAA', 'IIL', 'RDF', 'LYD', 'TPO', 'SEP', 'J1V',
                    'CCS', '48C', 'TPQ', 'ALN', 'TYR', 'VAL', 'FHO', 'CSS',
                    'APD', 'HY1', 'RE0', '95B', 'CSO', 'S4M', 'CSD', 'HIS',
                    'AC5', 'MDO', 'DAS', 'CGA', '48E', 'SAH', 'DBB', '0AR',
                    'LLP', 'TBG', '73C', 'FTY', 'XPR', 'LYH', 'SGB', '56A',
                    '7GA', 'ACB', 'ILO', 'D16', 'HTY', 'GLU', 'HGM', 'YOF',
                    'MIS', 'BET', 'MGN', '4L8', 'DNP', 'LED', 'LYS', 'MHO',
                    'CSX', 'GLJ', 'DBU', 'KCX', 'LEU', 'SFG', 'PSA', 'HOX',
                    '3FG', 'ASN', 'DLY', 'TA5', 'DSN', 'CME', 'CYS', 'MD5',
                    'FLC', 'PTH', 'TIH', 'DDE', 'FT6', '3MY', 'THR', 'OMY',
                    'CHG', 'NLG', 'PLG', 'SAM', 'PTM', 'PTR', 'DPR', 'TRQ',
                    '00C', 'FLT', 'K95', 'SCH', 'PHE', 'MET', 'ORN', 'OCS',
                    'MLE', 'DTY', 'LYO', '719', 'DAB', 'PCA']

        self.resolution = 0.1

    def loadMacroPDBInfo(self, macroPDB, rdkit_pkt=None):

        '''
        https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
        :param macroPDB: 蛋白质 大分子
        :return: 各种信息，位置
        '''
        mol = rdkit_pkt
        if mol is None:
            mol = Chem.MolFromPDBFile(macroPDB,removeHs=False)  #,removeHs=False ,proximityBonding=False
        conf = mol.GetConformer()
        info = []
        positions = []
        main_chain = []
        main_chain_set = ['N', 'O', 'C', 'CA']
        names = []
        reses = []
        names1 = []
        isaromatic = []

        hbd_hba_merge = []

        count = 0
        idx_map = {}
        H_flag = False

        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            s = atom.GetSymbol()
            if s=='H' :
                # print(s)
                H_flag=True
                continue

            monomerinfo = atom.GetMonomerInfo()
            name = monomerinfo.GetName().strip()
            res = atom.GetPDBResidueInfo()
            res_name = res.GetResidueName().strip()
            res_num = res.GetResidueNumber()
            pos = conf.GetAtomPosition(i)
            p = [pos.x, pos.y, pos.z]
            positions.append(p)
            ismain_chain = 0
            if name in main_chain_set:
                ismain_chain = 1
            names.append(s)
            if s == 'C':
                names1.append(1e9)
            else:
                names1.append(1)

            reses.append(res_name)
            aro = atom.GetIsAromatic()
            if aro:
                isaromatic.append(12)
            else:
                isaromatic.append(6)

            hnbrs = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum()==1]

            if len(hnbrs)>0 :
                hbd_hba_merge.append(3)
            else :
                hbd_hba_merge.append(2)

            idx_map[i] = count
            count+=1

        fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        acceptors = []
        donors = []
        hydrophobics = []

        hydrophobics.append(fdef.GetFeaturesForMol(mol, includeOnly="Hydrophobe"))
        hydrophobics.append(fdef.GetFeaturesForMol(mol, includeOnly="LumpedHydrophobe"))

        acceptors.append(fdef.GetFeaturesForMol(mol, includeOnly="Acceptor"))
        donors.append(fdef.GetFeaturesForMol(mol, includeOnly="Donor"))

        features = {
            "donors": donors,
            "acceptors": acceptors,
            "hydrophobics": hydrophobics
        }

        acc_id = [item.GetAtomIds()[0] for sublist in features["acceptors"] for item in sublist]
        donor_id = [item.GetAtomIds()[0] for sublist in features["donors"] for item in sublist]

        hba_hbd = [1] * len(reses)


        
        union = set(acc_id)&set(donor_id)

        for id in acc_id:
                if id in union:
                    continue
                atom = mol.GetAtomWithIdx(id)
                s = atom.GetSymbol()
                if s == 'H':
                    continue
                id_m = idx_map[id]
                if H_flag:
                    merge = hbd_hba_merge[id_m]

                    hba_hbd[id_m] = merge

                else:
                    hba_hbd[id_m] = 2

        for id in donor_id:
                if id in union:
                    continue
                atom = mol.GetAtomWithIdx(id)
                s = atom.GetSymbol()
                if s == 'H':
                    continue
                id_m = idx_map[id]

                if H_flag:

                    merge = hbd_hba_merge[id_m]

                    hba_hbd[id_m] = merge

                else:
                    hba_hbd[id_m] = 3


        for id in union:
            atom = mol.GetAtomWithIdx(id)
            s = atom.GetSymbol()
            if s == 'H':
                continue
            id_m = idx_map[id]
            hba_hbd[id_m] = 4

        paths = [list(path) for path in list(Chem.GetSymmSSSR(mol))]
        conf = mol.GetConformer()
        centroid = []
        for path in paths:
            if mol.GetAtomWithIdx(path[0]).GetIsAromatic():
                c = []

                for id in path:
                    p = conf.GetAtomPosition(id)

                    c.append([p.x, p.y, p.z])
                cen = np.mean(np.array(c), axis=0)
                centroid.append(cen.tolist())

        positions.extend(centroid)
        names.extend(['Centroid']*len(centroid))
        isaromatic.extend([0]*len(centroid))
        hba_hbd.extend([5]*len(centroid))

        return names, isaromatic, hba_hbd, positions




    def rotate(self, coords, rotMat, center=(0, 0, 0)):
        """
        Rotate a selection of atoms by a given rotation around a center
        """

        newcoords = coords - center
        return np.dot(newcoords, np.transpose(rotMat)) + center

    def recode(self, names, reses, ll1, contact,contact_scaffold, contact_start):

        new_names = []
        new_reses = []
        new_pos = []
        count = 0
        contact_new = []
        contact_scaffold_new = []
        new_contact_start = -1
        for i, l in enumerate(ll1):
            s = names[i]
            r = reses[i]

            if l[0] < 0 or l[0] >= 240:
                continue
            if l[1] < 0 or l[1] >= 240:
                continue
            if l[2] < 0 or l[2] >= 240:
                continue

            if s in self.symbol:

                new_names.append(self.symbol.index(s) + 1)
                new_reses.append(r)
                new_pos.append(l)

                if contact is not None:
                    contact_new.append(contact[i])
                if contact_scaffold is not None:
                    contact_scaffold_new.append(contact_scaffold[i])

                if contact_start is not None and i in contact_start:
                    new_contact_start = count
                count += 1

        return new_names, new_reses, np.array(new_pos), contact_new,contact_scaffold_new, new_contact_start

    def find_start(self,total_pos, query):
        dist = np.sqrt(np.sum(np.square(query[:, None] - total_pos[None, :]), axis=-1))

        indices = np.argmin(dist, axis=-1)
        return indices

    def select_k_contact(self, contact, topk=5):

        contact_idx = np.where(contact == 1)[0]

        np.random.shuffle(contact_idx)

        new_contact = np.zeros_like(contact)

        topk_l = min(len(contact_idx), topk)

        for i in range(topk_l):
            ind = contact_idx[i]

            new_contact[ind] = 1

        return new_contact

    def pocketCode(self, pdbPath, poc_pos, pocket_contact, lig_pos,center=None, rrot=None, rdkit_pkt=None):

        names, names1, reses, positions = self.loadMacroPDBInfo(pdbPath, rdkit_pkt)

        contact_start = self.find_start(np.array(positions),np.array([poc_pos]))

        contact_start = contact_start.tolist()

        min_poc_idx = self.find_start(np.array(positions),np.array(pocket_contact))

        contact = [0] * len(names)

        for i in range(len(min_poc_idx)):
            idx = min_poc_idx[i]
            contact[idx] = 1

        contact   = contact

        ### shape contact

        dist       = cdist(positions,lig_pos)

        dist_mask0 = np.sum(np.where(dist <= 4,1,0),axis=-1)

        dist_mask  = np.where(dist_mask0>0,1,0)

        contact_scaffold = dist_mask

        resolution = self.resolution

        length = int(24 / resolution)

        grid_c = (length - 1) / 2

        if center is None:

            center = np.mean(np.array(positions), axis=0)

        if rrot is not None:

            positions = self.rotate(np.array(positions), rrot, center=center)

        new_pdb_coords1 = (np.array(positions) - center) / resolution + np.array([grid_c, grid_c, grid_c])

        ll1 = np.rint(new_pdb_coords1).astype('int').tolist()

        reses = (np.array(names1)+np.array(reses)).tolist()

        new_names, new_reses, positions1, contact, contact_scaffold,contact_start = self.recode(names, reses, ll1, contact,contact_scaffold,contact_start)

        if contact_start == -1:

            raise "no posible contact"

        positions1 = positions1.tolist()

        
        if (len(new_names)>500) or (len(positions1)>500) or (len(contact)>500) or (len(contact_scaffold)>500):
            raise "Exceeding the maximum dimensional limit"

        final_symbol = new_names + [0] * (500 - len(new_names))

        final_reses = new_reses + [0] * (500 - len(new_reses))

        final_pos = positions1 + [[0, 0, 0]] * (500 - len(positions1))

        mask = [1] * len(new_names) + [0] * (500 - len(new_names))

        contact = contact + [0] * (500 - len(contact))

        contact_mask = np.where(np.array(final_symbol) == 1, 0, 1) * np.array(mask)

        contact = np.array(contact) * contact_mask

        contact_scaffold = contact_scaffold+[0]*(500-len(contact_scaffold))

        pos1 = final_pos[contact_start]

        if np.sum(contact)==0:

            raise 'no contact'

        contact_type = final_reses[contact_start]

        isaro = False

        if contact_type == 5:

            isaro = True

        new_contact = contact*2+contact_scaffold

        return final_symbol, final_reses, mask, np.array(final_pos), new_contact , pos1,isaro

    def pocketCodeNCI(self, pdbPath, center=None, pocket_contact=None, lig_pos=None):

        names, names1, reses, positions = self.loadMacroPDBInfo(pdbPath)

        contact = None
        if pocket_contact is not None:
            min_poc_idx = self.find_start(np.array(positions), np.array(pocket_contact))
            contact     = [0] * len(names)
            for i in range(len(min_poc_idx)):
                idx = min_poc_idx[i]
                contact[idx] = 1

        contact_scaffold = None
        if lig_pos is not None:
            dist = cdist(positions, lig_pos)
            dist_mask0 = np.sum(np.where(dist <= 4,1,0), axis=-1)
            contact_scaffold = np.where(dist_mask0>0,1,0)
            

        resolution = self.resolution
        length = int(24 / resolution)
        grid_c = (length - 1) / 2

        if center is None:
            center = np.mean(np.array(positions), axis=0)

        new_pdb_coords1 = (np.array(positions) - center) / resolution + np.array([grid_c, grid_c, grid_c])
        ll1 = np.rint(new_pdb_coords1).astype('int').tolist()
        reses = (np.array(names1)+np.array(reses)).tolist()

        new_names, new_reses, positions1, contact, contact_scaffold, _ = self.recode(names, reses, ll1, contact, contact_scaffold, None)
        positions1 = positions1.tolist()
        final_symbol = new_names + [0] * (500 - len(new_names))
        final_reses = new_reses + [0] * (500 - len(new_reses))
        final_pos = positions1 + [[0, 0, 0]] * (500 - len(positions1))
        mask = [1] * len(new_names) + [0] * (500 - len(new_names))

        contact = contact + [0] * (500 - len(contact))
        contact = np.array(contact) * np.array(mask)
        contact_scaffold = contact_scaffold+[0]*(500-len(contact_scaffold))

        return final_symbol, final_reses, mask, np.array(final_pos), center, np.array(contact), np.array(contact_scaffold)
