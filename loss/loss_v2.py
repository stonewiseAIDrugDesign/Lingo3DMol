import torch
import torch.nn.functional as F

def coords_NLL(gt_coords,pred_coords_prob):

    loss_func = torch.nn.NLLLoss()
    new_gt_coords = gt_coords
    pred_coords_prob = pred_coords_prob.permute(0,2,1)
    loss = loss_func(pred_coords_prob,new_gt_coords)
    return loss

def criterion(recon,target):

    reconstruction_function = torch.nn.CrossEntropyLoss()

    NLL = reconstruction_function(recon, target)

    return NLL


def dist_NLL1(gt_dist,pred_dist):

    loss_func = torch.nn.NLLLoss()
    pred_dist1 = pred_dist
    loss = loss_func(pred_dist1,gt_dist)

    return loss

def degree_NLL1(gt_degree,pred_degree):

    loss_func = torch.nn.NLLLoss()
    pred_dist1 = pred_degree#.permute(0,2,1)
    loss = loss_func(pred_dist1,gt_degree)

    return loss


def theta_NLL1(gt_theta,pred_theta):

    loss_func = torch.nn.NLLLoss()
    pred_dist1 = pred_theta
    loss = loss_func(pred_dist1,gt_theta)

    return loss

def loss_function(recon, target, label_coords, pred_pos, gt_dist, pred_dist, pred_dist_aux, gt_theta, pred_theta,
                   pred_theta_aux, gt_degree, pred_degree, pred_degree_aux):
    loss_func = torch.nn.NLLLoss()
    reconstruction_function = torch.nn.CrossEntropyLoss()

    # Coords NLL
    pred_coords_prob = pred_pos.permute(0, 2, 1)
    coords_nll = loss_func(pred_coords_prob, label_coords)

    # Criterion
    type_loss = reconstruction_function(recon, target)

    # Dist NLL
    dist_root_loss = loss_func(pred_dist, gt_dist)
    dist_root_loss_aux = loss_func(pred_dist_aux, gt_dist)

    # Degree NLL
    degree_loss = loss_func(pred_degree, gt_degree)
    degree_loss_aux = loss_func(pred_degree_aux, gt_degree)

    # Theta NLL
    theta_loss = loss_func(pred_theta, gt_theta)
    theta_loss_aux = loss_func(pred_theta_aux, gt_theta)

    total_loss = sum([coords_nll, dist_root_loss, theta_loss, degree_loss, type_loss,
                      dist_root_loss_aux, theta_loss_aux, degree_loss_aux])

    return total_loss, type_loss, coords_nll, dist_root_loss, dist_root_loss_aux, theta_loss, theta_loss_aux, degree_loss, degree_loss_aux