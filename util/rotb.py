from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts,NumHAcceptors,NumHDonors
import glob


def find_bond_groups(mol):
    """Find groups of contiguous rotatable bonds and return them sorted by decreasing size"""
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    rot_bond_set = set([mol.GetBondBetweenAtoms(*ap).GetIdx() for ap in rot_atom_pairs])
    rot_bond_groups = []
    while (rot_bond_set):
        i = rot_bond_set.pop()
        connected_bond_set = set([i])
        stack = [i]
        while (stack):
            i = stack.pop()
            b = mol.GetBondWithIdx(i)
            bonds = []
            for a in (b.GetBeginAtom(), b.GetEndAtom()):
                bonds.extend([b.GetIdx() for b in a.GetBonds() if (
                        (b.GetIdx() in rot_bond_set) and (not (b.GetIdx() in connected_bond_set)))])
            connected_bond_set.update(bonds)
            stack.extend(bonds)
        rot_bond_set.difference_update(connected_bond_set)
        rot_bond_groups.append(tuple(connected_bond_set))
    return tuple(sorted(rot_bond_groups, reverse=True, key=lambda x: len(x)))

def rotatable_bond(mol):
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    rot_bond_set = set([mol.GetBondBetweenAtoms(*ap).GetIdx() for ap in rot_atom_pairs])
    bond_groups = find_bond_groups(mol)
    largest_n_cont_rot_bonds = len(bond_groups[0]) if bond_groups else 0


    return largest_n_cont_rot_bonds


if __name__=='__main__':

    files = glob.glob('/home/bairong/data/pocket_data1/ligand/*sdf')

    for f in files:
        try:
            mol = Chem.SDMolSupplier(f)[0]

            rot = rotatable_bond(mol)

            if rot >5:
                print(rot,f,Chem.MolToSmiles(mol))
        except:
            continue
                # break


