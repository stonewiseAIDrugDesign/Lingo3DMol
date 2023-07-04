# This file is part of Lingo3DMol
#
# Lingo3DMol is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Lingo3DMol is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Lingo3DMol. If not, see <https://www.gnu.org/licenses/>.



import torch
def metric_acc(gt_dist,pred_dist,gt_theta,pred_theta,gt_degree,pred_degree):

    pred_dist_max = torch.max(pred_dist,dim=-1)[1] #
    dist_mask   = torch.where(gt_dist==0,0.0,1.0) # 只算有意义的token
    pred_dist_max1 = dist_mask*(pred_dist_max+9.0)/10.0
    pred_dist_max_ = pred_dist_max1.cpu().numpy()
    gt_dist = dist_mask*(gt_dist+9.0)/10.0
    gt_dist_ = gt_dist.cpu().numpy()


    mae_dist = abs(pred_dist_max1 - gt_dist)
    mae = torch.sum(mae_dist.float() * dist_mask)
    mae_dist = mae / torch.sum(dist_mask)

    pred_theta_max = torch.max(pred_theta,dim=-1)[1]  #
    theta_mask = torch.where(gt_theta == 0, 0.0, 1.0)  # 只算有意义的token
    pred_theta_max1 = theta_mask * pred_theta_max
    pred_theta_max1_ = pred_theta_max1.cpu().numpy()
    gt_theta_ = gt_theta.cpu().numpy()

    mae_theta = abs(pred_theta_max1 - gt_theta)
    mae = torch.sum(mae_theta.float() * theta_mask)
    mae_theta = mae / torch.sum(theta_mask)

    pred_degree_max = torch.max(pred_degree, dim=-1)[1]  #
    pred_degree_max1 = pred_degree_max #degree_mask *
    degree_mask = torch.ones_like(pred_degree_max1).cuda(device=pred_degree_max.device)

    mae_degree = abs(pred_degree_max1 - gt_degree)
    mae = torch.sum(mae_degree.float())
    mae_degree = mae / torch.sum(degree_mask)

    return mae_dist,mae_theta,mae_degree