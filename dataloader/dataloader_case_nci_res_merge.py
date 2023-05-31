import os
import traceback
import numpy as np
import torch.utils.data as data
import torch
from rdkit import Chem
from util.fragmol_frag_v0_2 import  FragmolUtil
from util.pocket_code_all import PocketCode

vocab = {'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'acc': 8, 'donor': 9}

class mydataset(data.Dataset):
    def __init__(self,fl_name):
        self.fnames = []
        self.ncis   = []
        self.lgds   = []
        for line in fl_name:
            lgdPath, pktNciPath, pdbPath = line.strip().split(',', 2)
            self.fnames.append(pdbPath.strip())
            self.ncis.append(pktNciPath.strip())
            self.lgds.append(lgdPath.strip())

        self.fragUtil = FragmolUtil()
        self.pocketCode = PocketCode()

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        while True:
            try:
                pdbPath, pktNciPath, lgdPath = self.fnames[index], self.ncis[index], self.lgds[index]

                pkt_nci_coords = None
                if os.path.isfile(pktNciPath):
                    pkt_nci_coords = np.load(pktNciPath, allow_pickle=True)

                lgd_coords = None
                if os.path.isfile(lgdPath):
                    lgd_coords = self.load_lgd_coords(lgdPath)

                type, residue, mask, new_coords,center, contact, contact_scaffold = self.pocketCode.pocketCodeNCI(pdbPath, center=None, pocket_contact=pkt_nci_coords, lig_pos=lgd_coords)
                if np.sum(contact) > 0:
                    contact = torch.FloatTensor(contact)
                else:
                    contact = None

                if np.sum(contact_scaffold) > 0:
                    contact_scaffold = torch.FloatTensor(contact_scaffold)
                else:
                    contact_scaffold = None

                if np.min(new_coords) < 0 or np.max(new_coords) >= 240:
                    print(f'new coords out of scale {np.min(new_coords)} {np.max(new_coords)}')
                    raise
                break

            except Exception as e:
                print(f'there is an exception {e}')
                index = np.random.randint(0, len(self.fnames))

        return torch.LongTensor(new_coords), torch.LongTensor(residue),torch.LongTensor(type), torch.FloatTensor(mask), torch.FloatTensor(center), index, contact, contact_scaffold

    def load_lgd_coords(self, ligand_file):
        lgd = Chem.SDMolSupplier(ligand_file, removeHs=True)[0]
        lgd = Chem.RemoveHs(lgd)
        lgd_coords = []

        for i in range(0, lgd.GetNumAtoms()):
            pos = lgd.GetConformer().GetAtomPosition(i)
            lgd_coords.append([pos.x, pos.y, pos.z])
        return lgd_coords
