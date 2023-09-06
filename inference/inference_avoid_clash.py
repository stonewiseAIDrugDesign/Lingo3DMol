# This file is part of Lingo3DMol
#
# Lingo3DMol is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Lingo3DMol is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Lingo3DMol. If not, see <https://www.gnu.org/licenses/>.



import os
os.sys.path.append('./')
import torch
import warnings
import time
import torch.nn as nn
from util.fragmol_frag_zyh import FragmolUtil
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem, rdGeometry
from model.transformer_v1_res_mp1 import TransformerModel as TransformerModel_contact
from model.transformer_v1_res_fac2 import TransformerModel
from dataloader.dataloader_case_nci_res_merge import mydataset as testdataset
from torch.utils.data import DataLoader
from rdkit import RDLogger
import time
from model.transformer_v1_res_mp1 import topkp_random
import argparse

RDLogger.DisableLog('rdApp.*')
os.environ["CUDA_DEVICE_ORDER"] =    "PCI_BUS_ID"
warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled= True
all_good_nums = 0

def bonusVSpenalty2(coc,pattern,semi_product):
    atom_pos = coc.get_atom_xyz(semi_product)
    return pattern[atom_pos[:,0],atom_pos[:,1],atom_pos[:,2]].sum() == atom_pos.shape[0]

def changepos(mol,savepath,center,issave=False):
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
    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        if atom_type == '*':
            mol.ReplaceAtom(atom.GetIdx(),Chem.Atom('H'))
    if issave:
        try:
            Chem.MolToMolFile(Chem.RemoveHs(mol),savepath)
        except Exception:
            return None


def changepos2(mol,center):
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
    return mol

def go_factory(factory_args,gudiecap,gudiepos,args):
    coords,residue,mask,atom_type,center,caption,contact_idx,contact_prob1,pattern,_ = factory_args
    sample_num = args.gen_frag_set
    recon, pred_pos,cap_mask, Value_prob = caption(coords=coords, type=atom_type,
                    residue=residue, src_mask=mask,
                    critical_anchor=contact_prob1,
                    contact_idx=contact_idx,
                    sample_num=sample_num, isTrain=args.isTrain, 
                    isMultiSample=args.isMultiSample,
                    USE_THRESHOLD=args.USE_THRESHOLD,
                    isGuideSample=args.isGuideSample,guidepath=[np.tile(gudiecap,(args.gen_frag_set,1)),np.tile(gudiepos,(args.gen_frag_set,1,1))],
                    start_this_step=(gudiecap>0).sum(),
                    OnceMolGen=args.OnceMolGen,frag_len=args.frag_len_add,tempture=args.tempture)
    captions = recon
    captions = captions.cpu().data.numpy()
    pred_pos = pred_pos.cpu().data.numpy()
    Value_prob = Value_prob.cpu().data.numpy()
    center = center.cpu().data.numpy()
    captions = captions*cap_mask
    pred_pos = pred_pos*cap_mask[:,:,np.newaxis]
    Value_prob = Value_prob*cap_mask
    return captions,pred_pos,center,Value_prob

def get_partial_to_warehouse(frag_level,factory_args,partial_product,args):
    coc = factory_args[8]
    pattern = coc.maskcoc
    fragUtil = FragmolUtil()
    all_captions = np.empty((0,100))
    all_pos = np.empty((0,100,3))
    all_center = np.empty((0,3))
    value_scores=[]
    anchor_scores_list=[]
    prod_time = 0
    uni_smi = set()
    one_line_product = []
    for frag in partial_product[0]:
        one_line_product.extend(frag)
    gudiecap = np.zeros((1,100))
    gudiecap[:,:len(one_line_product)]=np.array(one_line_product)[np.newaxis,:]
    gudiepos = np.zeros((1,100,3))
    one_line_coords = []
    for coo in partial_product[1]:
        one_line_coords.extend(coo)    
    if len(one_line_coords)!=0:
        gudiepos[:,:len(one_line_coords),:] = np.array(one_line_coords)[np.newaxis,:,:]
    
    factory_time = args.prod_time
    anchors = factory_args[0][factory_args[-1]==1]

    while all_captions.shape[0]<100 and prod_time<factory_time:
        print('level',frag_level,'prod time',prod_time,'frag num',all_captions.shape[0])
        captions,pred_pos,center,Value_prob = go_factory(factory_args,gudiecap,gudiepos,args)
        
        smiles, ss, moleculars = fragUtil.decode3d(captions, pred_pos)
        valid_index = []
        for j, smi in enumerate(ss):
            if smiles[j] is not None and moleculars[j] is not None:
                if smiles[j] not in uni_smi:
                    mol = changepos2(moleculars[j],center[j])
                    score = bonusVSpenalty2(coc,pattern,moleculars[j])
                    if not score:
                        continue
                    value = Value_prob[j].sum() / ((Value_prob[j]>0).sum()+1e-5)  
                    value_scores.append(value)  
                    distance = np.linalg.norm(anchors.cpu().numpy()[:,np.newaxis,:] - pred_pos[j][1:(captions[j]!=0).sum()],axis=-1)
                    anchor_score = (distance.min(axis=-1)<40).sum() / anchors.shape[0]
                    print("value",value,"anchor",anchor_score)
                    anchor_scores_list.append(anchor_score)
                    uni_smi.add(smiles[j])
                    valid_index.append(j)
        all_captions = np.append(all_captions,captions[valid_index],axis=0)
        all_pos = np.append(all_pos,pred_pos[valid_index],axis=0)
        all_center = np.append(all_center,center[valid_index],axis=0)
        prod_time+=1
    
    if len(value_scores)==0:
        return [],[]
    value_scores = np.array(value_scores)
    value_scores = (value_scores-value_scores.min())/(value_scores.max()-value_scores.min()+1e-5)
    anchor_scores_list = np.array(anchor_scores_list)
    anchor_scores_list = (anchor_scores_list-anchor_scores_list.min())/(anchor_scores_list.max()-anchor_scores_list.min()+1e-5)

    all_pos_indices = (value_scores+anchor_scores_list).argsort()[::-1]
    all_captions = all_captions[all_pos_indices]
    all_pos = all_pos[all_pos_indices]

    all_captions = all_captions[:args.gen_frag_set//5]
    all_pos = all_pos[:args.gen_frag_set//5]

    frag_new_cap = []
    frag_new_pos = []
    o_index = np.where(gudiecap==3)[-1]
    if len(o_index)==0:
        o_index=-1
    else:
        o_index=o_index[-1]

    for i in range(all_captions.shape[0]):
        cap = all_captions[i]
        pos = all_pos[i]
        if not args.OnceMolGen:
            index = np.argwhere(cap==2)
            if len(index):
                index = index[-1][-1]
            else:
                index = np.argwhere(cap==3)[-1][-1]
        else:
            try:
                index = np.argwhere(cap==2)[-1][-1]
            except:
                print('no good')
                continue
        frag_new_cap.append(cap[o_index+1:index+1])
        frag_new_pos.append(pos[o_index+1:index+1,:])
    return frag_new_cap,frag_new_pos

def write_product(partial_product):
    one_line_product = []
    for frag in partial_product[0]:
        one_line_product.extend(frag)
    gudiecap = np.zeros((1,100))
    gudiecap[:,:len(one_line_product)] = np.array(one_line_product)[np.newaxis,:]

    one_line_coords = []
    for coords in partial_product[1]:
        one_line_coords.extend(coords)
    gudiepos = np.zeros((1,100,3))
    gudiepos[:,:len(one_line_coords),:] = np.array(one_line_coords)[np.newaxis,:]

    fragUtil = FragmolUtil()
    smiles, ss, moleculars = fragUtil.decode3d(gudiecap, gudiepos)
    return smiles, ss, moleculars,gudiecap

def molecular_workflow(frag_level,warehouse,partial_product,fnames,factory_args,savedir,args):
    print('now level is', frag_level)
    global all_good_nums
    if len(partial_product[0]) and ((2 in partial_product[0][-1])):
        saveroot = savedir
        smiles, ss, moleculars,gudiecap =  write_product(partial_product)
        center = factory_args[4][0]
        name = fnames[0].strip().split('/')[-1].split('_post_final')[0]
        mol = changepos(moleculars[0],
                            saveroot + '/'+str(all_good_nums)+'_pred_'+str(frag_level)+'_'+ name + '.mol',
                            center.numpy(),issave=args.saveMol)
        
        all_good_nums+=1
        return 
    else:
        if frag_level+1>len(warehouse[0]):
            captions,pred_pos = get_partial_to_warehouse(frag_level,factory_args,partial_product,args)
            if len(captions)==0:
                return
            warehouse[0].append(captions)
            warehouse[1].append(pred_pos)
        for index in range(len(warehouse[0][frag_level])):
            print('index',str(index)+'/'+str(len(warehouse[0][frag_level])))
            frag = warehouse[0][frag_level][index]
            coords = warehouse[1][frag_level][index]
            partial_product[0].append(frag)
            partial_product[1].append(coords)
            molecular_workflow(frag_level+1,warehouse,partial_product,fnames,factory_args,savedir,args)
            partial_product[0].pop(-1)
            partial_product[1].pop(-1)
        warehouse[0].pop(-1)
        warehouse[1].pop(-1)

def validation(caption_contact,caption,testloader, testset, savedir, args):
    caption.eval()
    fnames = testset.fnames
    sample_num = args.gen_frag_set
    resolution = 0.1
    length = int(24 / resolution)
    caption.eval()
    warehouse = [[],[]]
    start_time = time.time()
    for i, (coords, residue, type, mask, center, index, contact_prob, contact_scaffold_prob) in enumerate(testloader):
            from inference.cube_collision_check import CollisionCheck
            coc = CollisionCheck(args.pocket_path_for_coc,args.coc_dis,center=center)
            break
    global all_good_nums
    while True:
        for i, (coords, residue, atom_type, mask, center, 
            index, contact_prob, contact_scaffold_prob) in enumerate(testloader):
            with torch.no_grad():
                coords = coords.cuda()
                residue = residue.cuda()
                mask = mask.cuda()
                atom_type = atom_type.cuda()
                index = index.repeat(sample_num)
                center = center.repeat(sample_num, 1)
                
                if contact_prob.shape[-1]==0 or contact_scaffold_prob.shape[-1]==0:
                    model_contact_prob, model_contact_scaffold_prob = caption_contact(coords=coords,
                                    residue=residue, atom_type=atom_type, src_mask=mask, isTrain=args.isTrain)
                if contact_prob.shape[-1]==0:
                    contact_prob = model_contact_prob
                else:
                    contact_prob = contact_prob.cuda()
                if contact_scaffold_prob.shape[-1]==0:
                    contact_scaffold_prob = model_contact_scaffold_prob
                else:
                    contact_scaffold_prob = contact_scaffold_prob.cuda()
                    
                contact_prob0 = torch.where(contact_prob > args.nci_thrs, 2, 0)
                contact_scaffold_prob1 = torch.where(contact_scaffold_prob > 0.9, 1, 0)
                contact_prob1 = contact_prob0 + contact_scaffold_prob1

                contact_prob = contact_prob.repeat(sample_num,1).cuda()
                residue_use      = residue.repeat(sample_num, 1)
                residue_mask    = torch.where(residue_use != 5, 1.0, 0.0)
                src_mask_repeat = mask.squeeze(1).repeat(sample_num,1)
                contact_prob    = contact_prob.masked_fill(src_mask_repeat == 0, 0)
                contact_prob    = contact_prob.masked_fill(residue_mask == 0, 0)
                contact_prob    = torch.softmax(contact_prob*5, dim=-1)
                contact_idx     = topkp_random(contact_prob, top_k=args.topk, top_p=0.9, thred=0.0)
                factory_args = [coords,residue,mask,atom_type,center,caption,contact_idx,contact_prob1,coc,contact_scaffold_prob1]
                molecular_workflow(0,warehouse,[[],[]],fnames,factory_args,savedir,args)
        if all_good_nums>args.gennums or time.time()-start_time>3600*args.max_run_hours:
            break

def get_pdb_files(ints):
    pdbs = []
    with open(ints,'r') as f:
        lines = f.readlines()
        for li in lines:
            pdbs.append(li.strip())
    return pdbs

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    cases_dude = get_pdb_files(args.input_list)
    caption_contact = TransformerModel_contact()
    path_model = args.contact_path
    print(path_model)
    dict_ = torch.load(path_model,map_location='cpu')

    caption_contact.load_state_dict(dict_,strict=False)
    caption_contact = nn.DataParallel(caption_contact)
    caption_contact.cuda()
    caption_contact.eval()

    caption = TransformerModel()
    caption.load_state_dict(torch.load(args.caption_path, map_location='cpu'))
    caption = nn.DataParallel(caption)
    caption.cuda()
    caption.eval()

    args.pocket_path_for_coc = cases_dude[0].split(',')[-1]
    for i,case in enumerate(cases_dude):
        print(case,'-'*100)

        cases = [case]
        name = case.split('/')[-1]
        print(f'{i} {case} .........................................................')
        savedir = f'{args.savedir}{args.cuda}/{name}'
        print(savedir)
        if args.saveMol:
            os.system(f'rm -rf {savedir}')
            os.makedirs(savedir, exist_ok=True)

        testset = testdataset(cases)

        testloader = DataLoader(dataset=testset,
                                batch_size=1,
                                shuffle=False,pin_memory=True, num_workers=0)

        print("Prep data done", len(testloader))
        validation(caption_contact,caption, testloader, testset, savedir, args)
        

if __name__=='__main__':

    import pytz
    from datetime import datetime

    start = time.time()
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--savedir',type=str, help='savepath')
    parser.add_argument('--contact_path', type=str,default='checkpoint/contact.pkl')
    parser.add_argument('--caption_path', type=str,default='checkpoint/gen_mol.pkl')
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--coc_dis', type=float, default=2.5)
    parser.add_argument('--nci_thrs', type=float, default=0.7)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--max_run_hours', type=int)
    parser.add_argument('--gennums', type=int)
    parser.add_argument('--cuda_list', type=int,nargs='+')
    parser.add_argument('--input_list', type=str)
    parser.add_argument('--saveMol', action='store_true', default=True)
    parser.add_argument('--isTrain', action='store_true')
    parser.add_argument('--USE_THRESHOLD', action='store_true', default=True)
    parser.add_argument('--isMultiSample', action='store_true', default=True)
    parser.add_argument('--isGuideSample', action='store_true', default=True)
    parser.add_argument('--OnceMolGen', action='store_true')
    parser.add_argument('--gen_frag_set', type=int, default=1)
    parser.add_argument('--prod_time', type=int, default=1)
    parser.add_argument('--tempture', type=float, default=1.0)
    parser.add_argument('--frag_len_add', type=int, default=0)
    args = parser.parse_args()
    import logging
    if args.cuda == str(args.cuda):
        logging.basicConfig(filename='logs.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
        tz = pytz.timezone('Asia/Shanghai')    
        now = datetime.now(tz)
        for arg, value in args.__dict__.items():
            logging.info('%s: %s', arg, value)
    main(args)

    if args.cuda == str(args.cuda):
        with open(args.save_time_path,'a') as f:
            f.write(args.savedir+','+str(time.time()-start)+'\n')
        print("alltime is",time.time()-start)
        now = datetime.now(tz)
