import scipy.linalg as sl
import torch.nn as nn

# from utils.Quaternions_torch import qeuler
from utils.common import *


def mae_joints(poses_target, poses_predicted):
    return np.mean(np.linalg.norm(np.reshape(poses_target - poses_predicted,
                                             (len(poses_predicted), -1)), axis=1, ord=1))


def frechet_gesture_distance(features_target, features_predicted):
    '''
    :param features_target: numpy array of shape (N, 32), where N is the number of samples
    :param features_predicted: numpy array of shape (N, 32), where N is the number of samples
    :return: a non-negative number denoting the Frechet Gesture Distance
    '''
    mean_features_target = np.mean(features_target, axis=0)
    cov_features_target = np.cov(np.transpose(features_target - mean_features_target))
    mean_features_predicted = np.mean(features_predicted, axis=0)
    cov_features_predicted = np.cov(np.transpose(features_predicted - mean_features_predicted))
    fgd = np.power(np.linalg.norm(mean_features_target - mean_features_predicted), 2.) +\
        np.trace(cov_features_target + cov_features_predicted -
                 2. * sl.sqrtm(np.matmul(cov_features_target, cov_features_predicted)))
    return fgd


def quat_angle_loss(quats_pred, quats_target, quat_valid_idx, V, D,
                    lower_body_start=15, upper_body_weights=1., drift_len=20):
    quats_pred = quats_pred.reshape(-1, quats_pred.shape[1], V, D)
    quats_target = quats_target.reshape(-1, quats_target.shape[1], V, D)
    euler_pred = qeuler(quats_pred.contiguous(), order='yzx', epsilon=1e-6)
    euler_target = qeuler(quats_target.contiguous(), order='yzx', epsilon=1e-6)
    # L1 loss on angle distance with 2pi wrap-around
    angle_distances = torch.remainder(euler_pred[:, 1:] - euler_target[:, 1:] + np.pi, 2 * np.pi) - np.pi
    angle_distances[:, :, :lower_body_start] = upper_body_weights * angle_distances[:, :, :lower_body_start]
    angle_derv_distances = torch.zeros_like(angle_distances)
    for idx in range(1, drift_len):
        angle_derv_distances[:, idx - 1:] += euler_pred[:, idx:] - euler_pred[:, :-idx] -\
                                             euler_target[:, idx:] + euler_target[:, :-idx]
        # angle_derv_distances += euler_pred[:, 1:] - euler_pred[:, :-1] - euler_target[:, 1:] + euler_target[:, :-1]
    angle_derv_distances[:, :, :lower_body_start] =\
        upper_body_weights * angle_derv_distances[:, :, :lower_body_start]
    return torch.mean(torch.abs(angle_distances)), torch.mean(torch.abs(angle_derv_distances))

    # angle_distances = torch.abs(
    #     torch.remainder(euler_pred - euler_target + np.pi, 2 * np.pi) - np.pi).sum(-1)
    # angle_distances = upper_body_weights * (angle_distances[:, :, :lower_body_start].sum(-1)) +\
    #                   angle_distances[:, :, lower_body_start:].sum(-1)
    # angle_derv_distances = torch.abs(
    #     euler_pred[:, 1:] - euler_pred[:, :-1] - euler_target[:, 1:] + euler_target[:, :-1]).sum(-1)
    # angle_derv_distances = upper_body_weights * (angle_derv_distances[:, :, :lower_body_start].sum(-1)) +\
    #                        angle_derv_distances[:, :, lower_body_start:].sum(-1)
    # row_sums = quat_valid_idx.sum(1, keepdim=True) * D * V
    # row_sums[row_sums == 0.] = 1.
    # return torch.mean((quat_valid_idx * angle_distances).sum(-1) / row_sums),\
    #     torch.mean((quat_valid_idx[:, 1:] * angle_derv_distances).sum(-1) / row_sums)
