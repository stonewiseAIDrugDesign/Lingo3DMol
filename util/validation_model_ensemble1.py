# This file is part of Lingo3DMol
#
# Lingo3DMol is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Lingo3DMol is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Lingo3DMol. If not, see <https://www.gnu.org/licenses/>.



import os
os.sys.path.append(os.pardir)
import torch
import warnings
import time
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from loss.loss_v2 import loss_function
from util.fragmol_frag_v0_2 import FragmolUtil as MolUtil
from loss.metric4 import metric_acc

from util.sas_qed import calculateScore
from rdkit.Chem import QED

os.environ["CUDA_DEVICE_ORDER"] =    "PCI_BUS_ID"
device_ids=[0,1,2]
warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled= True
epochs = 200
vocab = ['C','N','O','S' ,'P','F','Cl','Br','I','total']
lambda_ = 10
vocab2 = {'C':1,'N':2,'O':3,'S':4,'P':5,'F':6,'Cl':7,'Br':8,'I':9}

def nonC_rate(mol):
    total = 0
    nonC = 0
    non = ['C']
    for i in range(mol.GetNumAtoms()):
        s = mol.GetAtomWithIdx(i).GetSymbol()
        if s in vocab2.keys():
            total+=1
            if s not in non:
               nonC+=1

    return nonC/total


def criterion(recon,target):
    # label = quantized.max(dim=1)[1]

    reconstruction_function = nn.CrossEntropyLoss()
    NLL = reconstruction_function(recon, target)
    return NLL
def dice_batch_single(vox_ref, vox_g):

    vox_g = vox_g
    vox_ref = vox_ref
    sum_ref = torch.sum(torch.square(vox_ref), dim=-1)
    sum_gt = torch.sum(torch.square(vox_g), dim=-1)
    intersection = vox_ref * vox_g
    intersection_score = torch.sum(intersection, dim=-1)
    total_score = sum_ref + sum_gt - intersection_score
    score = intersection_score / total_score
    score1 = score.reshape(-1)

    return score1.cpu().numpy().tolist()

# fragUtil = MolUtil()
fragUtil = MolUtil()

def validation(caption_contact,caption,testloader,testset,savedir,rank):
            if rank==0:
                f = open(savedir+'/pred','w')
                ff = open(savedir+'/smiles.smi','w')

            losses = []
            caption.eval()
            total = 0
            count = 0
            unique = set()
            nonC = []
            scores_finger = []
            fnames = testset.fnames
            dist_metrics = []
            theta_metrics = []
            degree_metrics = []
            sas = []
            qed = []


            for i, (code, residue,contact,coords, atom_type,mask,tgt_coords,smi_map, smi_map_n1, smi_map_n2,theta,dist,degree,finger,center,index) in enumerate(testloader):

                    with torch.no_grad():

                        t1 = time.time()

                        code = Variable(code.cuda())
                        coords = coords.cuda()
                        residue = residue.cuda()
                        mask = mask.cuda()
                        atom_type = atom_type.cuda()
                        dist = dist.cuda()
                        degree = degree.cuda()
                        contact = contact.cuda()

                        finger = finger.repeat(2,1)
                        index = index.repeat(2)


                        targets = code[:,1:].reshape(-1)
                        tgt_coords = tgt_coords.cuda()
                        label_coords = tgt_coords[:, 1:, :]
                        gt_coords = tgt_coords

                        smi_map = smi_map.cuda()
                        smi_map_n1 = smi_map_n1.cuda()
                        smi_map_n2 = smi_map_n2.cuda()
                        theta = theta.cuda()

                        recon, pred_pos,dist_pred,dist_pred_aux,theta_pred,theta_pred_aux,degree_pred ,degree_pred_aux = caption(coords=coords, residue=residue, atom_type=atom_type, critical_anchor=contact, src_mask= mask,captions=code,gt_coords=gt_coords,smi_map=smi_map, smi_map_n1=smi_map_n1, smi_map_n2=smi_map_n2,isTrain=True)

                        label_ind = torch.where(code[:, 1:] != 0)
                        targets = code[:, 1:][label_ind].reshape(-1)
                        dist = dist[:, 1:][label_ind]
                        theta = theta[:, 1:][label_ind]
                        degree = degree[:, 1:][label_ind]
                        label_coords = label_coords[label_ind]

                        recon = recon[:, :-1]
                        recon = recon[label_ind]
                        recon = recon.reshape(-1, 76)
                        pred_pos = pred_pos[label_ind]
                        dist_pred = dist_pred[label_ind]
                        dist_pred_aux = dist_pred_aux[label_ind]
                        theta_pred = theta_pred[label_ind]
                        theta_pred_aux = theta_pred_aux[label_ind]
                        degree_pred = degree_pred[label_ind]
                        degree_pred_aux = degree_pred_aux[label_ind]


                        loss,type_loss,coords_nll,dist_loss1,dist_loss1_aux,theta_loss,theta_loss_aux,degree_loss,degree_loss_aux= loss_function(recon,targets,label_coords,pred_pos,dist,dist_pred,dist_pred_aux,theta,theta_pred,theta_pred_aux,degree,degree_pred,degree_pred_aux)

                        losses.append(loss.item())
                        dist_metric,theta_metric,degree_metric = metric_acc(dist,dist_pred_aux,theta,theta_pred_aux,degree,degree_pred_aux)
                        dist_metrics.append(dist_metric.item())
                        theta_metrics.append(theta_metric.item())
                        degree_metrics.append(degree_metric.item())
                        if rank==0:
                            print(loss.item(),type_loss.item(), coords_nll.item(),dist_loss1.item(),theta_loss.item(),dist_metric.item(),theta_metric.item(),degree_metric.item())



                        contact_prob,contact_scaffold_prob   = caption_contact(coords=coords, residue=residue, atom_type=atom_type, src_mask=mask)

                        contact_prob0 = torch.where(contact_prob>0.9,2,0)

                        contact_scaffold_prob1 = torch.where(contact_scaffold_prob>0.9,1,0)

                        contact_prob1 = contact_prob0 + contact_scaffold_prob1

                        recon,pred_pos = caption(coords=coords,atom_type=atom_type,residue=residue,src_mask = mask,critical_anchor=contact_prob1,contact=contact_prob,isTrain=False)

                        captions = recon

                        captions = captions.cpu().data.numpy()

                        pred_pos = pred_pos.cpu().data.numpy()

                        smiles,ss,_ = fragUtil.decode3d(batch_codes=captions,positions=pred_pos)

                        total += 1

                        finger_ref = []

                        finger = finger.numpy()

                        finger_pred = []

                        center = center.cpu().data.numpy()

                        for j,smi in enumerate(ss):
                            total += 1

                            print(total,smi)
                            if rank==0:
                                f.write(str(total)+','+str(smi)+'\n')
                                f.flush()
                            if smiles[j] is not None:
                                finger_ref.append(finger[j])
                                count += 1
                                unique.add(smiles[j])
                                nonC.append(nonC_rate(Chem.MolFromSmiles(smiles[j])))
                                print(smiles[j], count, total)
                                mol2 = Chem.MolFromSmiles(smiles[j])

                                sas_single = calculateScore(mol2)
                                sas.append(sas_single)
                                qed_single = QED.qed(mol2)
                                qed.append(qed_single)

                                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
                                ss = fp1.ToBitString()
                                a = np.array([int(i) for i in ss])

                                finger_pred.append(a)
                                if rank==0:
                                    f.write(smiles[j] + ',' + str(count) + ',' + str(total) + '\n')
                                    f.flush()
                                path = str(fnames[index[j]])
                                name = path.strip().split('/')[-1].split('_post_final')[0]
                                if rank==0:
                                    ff.write(smiles[j] + ',' + name + '\n')

                        finger_ref = torch.FloatTensor(finger_ref).cuda()
                        finger_pred = torch.FloatTensor(finger_pred).cuda()
                        scores = dice_batch_single(finger_ref, finger_pred)

                        scores_finger.extend(scores)

            if rank==0:

                print(np.mean(losses),count/total,len(unique)/total,np.mean(nonC),np.mean(scores_finger),np.mean(dist_metrics),np.mean(theta_metrics),np.mean(degree_metrics),np.mean(qed),np.mean(sas))

            if rank==0:

                f.close()

                ff.close()

            return np.mean(losses),count/total,len(unique)/total,np.mean(nonC),np.mean(scores_finger),np.mean(dist_metrics),np.mean(theta_metrics),np.mean(degree_metrics),np.mean(qed),np.mean(sas)