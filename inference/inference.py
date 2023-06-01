import os
import sys
os.sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
import torch
import warnings
import time
import torch.nn as nn
from util.fragmol_frag_v0_2 import FragmolUtil
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem, rdGeometry
from model.transformer_v1_res_mp1 import TransformerModel as TransformerModel_contact
from model.transformer_v1_res import TransformerModel
from dataloader.dataloader_case_nci_res_merge import mydataset as testdataset
from torch.utils.data import DataLoader
from rdkit import RDLogger
import glob
from rdkit.Chem import QED
import argparse

RDLogger.DisableLog('rdApp.*')
os.environ["CUDA_DEVICE_ORDER"] =    "PCI_BUS_ID"
warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled= True

def changepos(mol,savepath,center):

    mol = Chem.RWMol(mol)
    conf = mol.GetConformer()
    resolution = 0.1
    length = int(24 / resolution)
    grid_c = (length - 1) / 2

    for i in range(mol.GetNumAtoms()):

        pos = conf.GetAtomPosition(i)

        pos1 = (np.array([pos])-grid_c)*resolution+center

        pos1 = pos1[0]

        p = rdGeometry.Point3D(float(pos1[0]), float(pos1[1]), float(pos1[2]))

        conf.SetAtomPosition(i, p)

    Chem.MolToMolFile(mol,savepath)


def validation(caption_contact, caption,testloader, testset, savedir):
    fragUtil = FragmolUtil()

    saveroot = savedir
    f = open(savedir + '/pred', 'w')
    ff = open(savedir + '/smiles.smi', 'w')

    losses = []
    caption.eval()
    total = 0
    count = 0
    unique = set()

    fnames = testset.fnames

    sample_num = 100
    epochs=500

    resolution = 0.1
    length = int(24 / resolution)
    grid_c = (length - 1) / 2

    for e in range(epochs):
        for i, (coords, residue, atom_type, mask, center, index, contact_prob, contact_scaffold_prob) in enumerate(testloader):
                with torch.no_grad():

                    t1 = time.time()

                    coords = coords.cuda()
                    residue = residue.cuda()
                    mask = mask.cuda()
                    atom_type = atom_type.cuda()

                    index = index.repeat(sample_num)
                    center = center.repeat(sample_num, 1)


                    model_contact_prob, model_contact_scaffold_prob = caption_contact(coords=coords,
                                    residue=residue, atom_type=atom_type, src_mask=mask, isTrain=False)
                    if len(contact_prob[0]) == 0:
                        contact_prob = model_contact_prob
                    if len(contact_scaffold_prob[0]) == 0:
                        contact_scaffold_prob = model_contact_scaffold_prob

                    #print('contact_prob:', contact_prob, np.argwhere(contact_prob > 0))
                    #print('contact_scaffold_prob:', contact_scaffold_prob, np.argwhere(contact_scaffold_prob > 0))
                    contact_prob0 = torch.where(contact_prob > 0.9, 2, 0)
                    contact_scaffold_prob1 = torch.where(contact_scaffold_prob > 0.9, 1, 0)

                    contact_prob1 = contact_prob0 + contact_scaffold_prob1

                    recon, pred_pos = caption(coords=coords, atom_type=atom_type,
                                    residue=residue, src_mask=mask,
                                    critical_anchor=contact_prob1,
                                    contact=contact_prob,
                                    sample_num=sample_num, isTrain=False, isMultiSample=True)

                    captions = recon
                    captions = captions.cpu().data.numpy()
                    pred_pos = pred_pos.cpu().data.numpy()
                    center = center.cpu().data.numpy()
                    smiles, ss, moleculars = fragUtil.decode3d(captions, pred_pos)

                    finger_pred = []
                    uni_smi = set()
                    qeds = []
                    for j, smi in enumerate(ss):
                        total += 1

                        print('FSMILES;', total, ';', ''.join([s for s in smi]), ';',
                              [str(a) + ':' + t for a, t in enumerate(smi)])
                        f.write(str(total) + ',' + str(smi) + '\n')
                        f.flush()
                        if smiles[j] is not None and moleculars[j] is not None:

                            uni_smi.add(smiles[j])

                            count += 1

                            if smiles[j] in unique:
                                continue
                            else:
                                unique.add(smiles[j])

                            print(smiles[j], count, total)

                            mol2 = Chem.MolFromSmiles(smiles[j])

                            qeds.append(QED.qed(mol2))

                            print(qeds[-1])

                            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
                            ss = fp1.ToBitString()
                            a = np.array([int(i) for i in ss])
                            finger_pred.append(a)

                            f.write(smiles[j] + ',' + str(count) + ',' + str(total) + '\n')
                            f.flush()
                            path = str(fnames[index[j]])
                            name = path.strip().split('/')[-1].split('_post_final')[0]
                            ff.write(smiles[j] + ',' + name + ',' + str(e) + '_' + str(i) + '_' + str(
                                j) + '_pred_' + name + '.pdb' + '\n')
                            ff.flush()

                            path = str(fnames[index[j]])
                            name = path.strip().split('/')[-1]
                            savepath = os.path.join(saveroot, str(e) + '_' + str(i) + '_' + str(j) + '_orig_' + name)
                            os.system(f'cp {path} {savepath}')
                            # changepos(moleculars[j], saveroot +  name + '_3d.pdb', center[j])
                            changepos(moleculars[j],
                                      saveroot + str(e) + '_' + str(i) + '_' + str(j) + '_pred_' + name + '.mol',
                                      center[j])

def get_pdb_files():
    pdbs = []
    for l in sys.stdin:
        pdbs.append(l.strip())
    return pdbs

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    savepath = args.save_path

    #cases_dude = glob.glob('/home/bairong/project/PocketBased/udf_pocketbase_dude/pocket/*_pocket.pdb')
    cases_dude = get_pdb_files()

    for i,case in enumerate(cases_dude):

        print(case,'-'*100)


        caption_contact = TransformerModel_contact()
        path_model = 'checkpoint/contact.pkl'
        print(path_model)
        dict_ = torch.load(path_model,map_location='cpu')

        caption_contact.load_state_dict(dict_,strict=False)
        caption_contact = nn.DataParallel(caption_contact)
        caption_contact.cuda()
        caption_contact.eval()

        caption = TransformerModel()
        #caption.load_state_dict(torch.load('/home/jovyan/project/v9_genmol_exp/pocketbased_genmol/checkpoint/gen_mol.pkl', map_location='cpu'))
        caption.load_state_dict(torch.load('checkpoint/gen_mol.pkl', map_location='cpu'))
        caption = nn.DataParallel(caption)
        caption.cuda()
        caption.eval()


        cases = [case]
        name = case.split('/')[-1]
        print(f'{i} {case} .........................................................')
        savedir = os.path.join(savepath, name)
        print(savedir)
        os.system(f'rm -rf {savedir}')
        os.makedirs(savedir, exist_ok=True)

        testset = testdataset(cases)

        testloader = DataLoader(dataset=testset,
                                batch_size=1,
                                shuffle=False,pin_memory=True, num_workers=0)

        print("Prep data done", len(testloader))

        validation(caption_contact,caption, testloader, testset, savedir)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Script for protein-ligand interaction prediction")
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device number')
    parser.add_argument('--save_path', type=str, default='output/', help='save directory path')
    #sample num
    parser.add_argument('--sample_num', type=int, default=100, help='sample num')
    args = parser.parse_args()

    main(args)
    
