# This file is part of Lingo3DMol
#
# Lingo3DMol is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Lingo3DMol is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Lingo3DMol. If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import torch
from rdkit import Chem

class CollisionCheck():
    def __init__(self, protein_path,distance,seed_conf=None,center=None,path=None):
        if not seed_conf is None:
                self.seed_center = np.mean(seed_conf, axis=0)
        elif not center is None:
            self.seed_center = np.array(center)[0]
        if path is None:
            mol = Chem.MolFromPDBFile(protein_path,removeHs=True)
            atom_pos = self.get_atom_xyz(mol)
            center = self.seed_center
            array = self.get_grid_file(atom_pos,center,resolution=0.1,distance=distance)
            maskcoc,_ = self.voxelize_xtb1(array,center=center,resolution=0.1)
            maskcoc = torch.from_numpy(maskcoc)
        else:
            maskcoc = torch.from_numpy(np.load(path))
        self.maskcoc = maskcoc.cuda()

    def get_atom_xyz(self,mol):
        conformer = mol.GetConformer()
        atom_pos = np.empty((0,3))
        for atom in mol.GetAtoms():
            atom_type = atom.GetSymbol()
            if atom_type=='*':
                continue
            atom_index = atom.GetIdx()
            atom_position = conformer.GetAtomPosition(atom_index)
            atom_pos = np.append(atom_pos,np.expand_dims(np.array([atom_position.x,atom_position.y,atom_position.z]),0),axis=0)
        return atom_pos

    def voxelize_xtb1(self,density_coords_,resolution=0.5,center=None):
        length = int(24 / resolution)
        grid_c = (length - 1) / 2
        grid_d = np.ones((length, length, length))
        d = np.rint((density_coords_-center)/ resolution) + np.array([grid_c, grid_c, grid_c])
        d = d.astype('int').T
        idx1 = set(np.where((0 <= d[0]) & (d[0] <= length - 1))[0])
        idx2 = set(np.where((0 <= d[1]) & (d[1] <= length - 1))[0])
        idx3 = set(np.where((0 <= d[2]) & (d[2] <= length - 1))[0])
        indice = list(set((idx1 & idx2 & idx3)))
        grid_d[d[0][indice], d[1][indice], d[2][indice]] = 0.0  # density_val1[indice]  density_value[indice]
        return grid_d,center

    def get_grid_file(self,atom_pos,center,resolution=0.5,distance=2):
        atom_dis = (atom_pos-center)//resolution
        new_all_atoms = (center+resolution*atom_dis)
        array = np.empty((0,3),dtype=np.float)
        for index in range(new_all_atoms.shape[0]):
            print(index)
            bigger_num = distance*1.2*2//resolution//2
            offsets = np.arange(-bigger_num, bigger_num + 1)*resolution
            new_xyz = new_all_atoms[index]
            coors = new_xyz + np.transpose(np.meshgrid(offsets, offsets, offsets), (1, 2, 3, 0)).reshape(-1, 3)
            dis = np.linalg.norm(coors - new_xyz, axis=-1)
            print(coors[dis<=distance].shape,coors.shape)
            array = np.append(array,coors[dis<=distance],axis=0)
        return array