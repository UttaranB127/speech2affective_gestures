# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import scipy.ndimage.filters
import utils.common as common

from utils.Quaternions import Quaternions
from utils.Quaternions_torch import *


class MoCapDataset:
    def __init__(self, joints_to_model, joint_parents_all, joint_parents_to_model, joints_left, joints_right):
        self.joints_to_model = joints_to_model
        self.joints_parents_all = joint_parents_all
        self.joints_parents = joint_parents_to_model
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.has_children_all = np.zeros_like(self.joint_parents_all)
        for j in range(len(self.joint_parents_all)):
            self.has_children_all[j] = np.isin(j, self.joint_parents_all)
        self.has_children = np.zeros_like(self.joint_parents)
        for j in range(len(self.joint_parents)):
            self.has_children[j] = np.isin(j, self.joint_parents)

    @staticmethod
    def has_children(joint, joint_parents):
        return np.isin(joint, joint_parents)

    @staticmethod
    def forward_kinematics(rotations, root_positions, parents, offsets):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
         -- parents: (J) numpy array where each element i contains the parent of joint i.
         -- offsets: (N, J, 3) tensor containing the offset of each joint in the batch.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = offsets.expand(rotations.shape[0], rotations.shape[1],
                                          offsets.shape[-2], offsets.shape[-1]).contiguous()

        # Parallelize along the batch and time dimensions
        for i in range(offsets.shape[-2]):
            if parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[parents[i]], expanded_offsets[:, :, i]) \
                                       + positions_world[parents[i]])
                if MoCapDataset.has_children(i, parents):
                    rotations_world.append(qmul(rotations_world[parents[i]], rotations[:, :, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

    @staticmethod
    def load_bvh(file_name, channel_map=None,
                 start=None, end=None, order=None, world=False):
        """
        Reads a BVH file and constructs an animation

        Parameters
        ----------
        file_name: str
            File to be opened

        channel_map: Dict
            Mapping between the coordinates x, y, z and
            the positions X, Y, Z in the bvh file

        start : int
            Optional Starting Frame

        end : int
            Optional Ending Frame

        order : str
            Optional Specifier for joint order.
            Given as string E.G 'xyz', 'zxy'

        world : bool
            If set to true euler angles are applied
            together in world space rather than local
            space

        Returns
        -------

        (animation, joint_names, frame_time)
            Tuple of loaded animation and joint names
        """

        if channel_map is None:
            channel_map = {'Xrotation': 'x', 'Yrotation': 'y', 'Zrotation': 'z'}
        f = open(file_name, 'r')

        i = 0
        active = -1
        end_site = False

        names = []
        orients = Quaternions.id(0)
        offsets = np.array([]).reshape((0, 3))
        parents = np.array([], dtype=int)

        for line in f:

            if 'HIERARCHY' in line:
                continue
            if 'MOTION' in line:
                continue

            root_match = re.match(r'ROOT (\w+)', line)
            if root_match:
                names.append(root_match.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if '{' in line:
                continue

            if '}' in line:
                if end_site:
                    end_site = False
                else:
                    active = parents[active]
                continue

            offset_match = re.match(r'\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)', line)
            if offset_match:
                if not end_site:
                    offsets[active] = np.array([list(map(float, offset_match.groups()))])
                continue

            channel_match = re.match(r'\s*CHANNELS\s+(\d+)', line)
            if channel_match:
                channels = int(channel_match.group(1))
                if order is None:
                    channel_is = 0 if channels == 3 else 3
                    channel_ie = 3 if channels == 3 else 6
                    parts = line.split()[2 + channel_is:2 + channel_ie]
                    if any([p not in channel_map for p in parts]):
                        continue
                    order = ''.join([channel_map[p] for p in parts])
                continue

            joint_match = re.match('\s*JOINT\s+(\w+)', line)
            if joint_match:
                names.append(joint_match.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if 'End Site'.lower() in line.lower():
                end_site = True
                continue

            frame_match = re.match('\s*Frames:\s+(\d+)', line)
            if frame_match:
                if start and end:
                    frame_num = (end - start) - 1
                else:
                    frame_num = int(frame_match.group(1))
                joint_num = len(parents)
                positions = offsets[np.newaxis].repeat(frame_num, axis=0)
                rotations = np.zeros((frame_num, len(orients), 3))
                continue

            frame_match = re.match('\s*Frame Time:\s+([\d\.]+)', line)
            if frame_match:
                frame_time = float(frame_match.group(1))
                continue

            if (start and end) and (i < start or i >= end - 1):
                i += 1
                continue

            data_match = line.strip().split(' ')
            if data_match:
                data_block = np.array(list(map(float, data_match)))
                N = len(parents)
                fi = i - start if start else i
                if channels == 3:
                    positions[fi, 0:1] = data_block[0:3]
                    rotations[fi, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    positions[fi, :] = data_block[:, 0:3]
                    rotations[fi, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[fi, 0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rotations[fi, 1:] = data_block[:, 3:6]
                    positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
                else:
                    raise Exception('Too many channels! {}'.format(channels))

                i += 1

        f.close()

        rotations = qfix(Quaternions.from_euler(np.radians(rotations), order=order, world=world).qs)
        positions = MoCapDataset.forward_kinematics(torch.from_numpy(rotations).cuda().float().unsqueeze(0),
                                                    torch.from_numpy(positions[:, 0]).cuda().float().unsqueeze(0),
                                                    parents,
                                                    torch.from_numpy(offsets).cuda().float()).squeeze().cpu().numpy()
        orientations, _ = Quaternions(rotations[:, 0]).angle_axis()
        return names, parents, offsets, positions, rotations, 1. / frame_time

    @staticmethod
    def traverse_hierarchy(hierarchy, joint_names, joint_offsets, joint_parents,
                           joint, metadata, tabs, rot_string):
        if joint > 0:
            metadata += '{}JOINT {}\n{}{{\n'.format(tabs, joint_names[joint], tabs)
            tabs += '\t'
            metadata += '{}OFFSET {:.6f} {:.6f} {:.6f}\n'.format(tabs,
                                                                 joint_offsets[joint][0],
                                                                 joint_offsets[joint][1],
                                                                 joint_offsets[joint][2])
            metadata += '{}CHANNELS 3 {}\n'.format(tabs, rot_string)
        while len(hierarchy[joint]) > 0:
            child = hierarchy[joint].pop(0)
            metadata, tabs = MoCapDataset.traverse_hierarchy(hierarchy, joint_names,
                                                             joint_offsets, joint_parents,
                                                             child, metadata, tabs, rot_string)
        if MoCapDataset.has_children(joint, joint_parents):
            metadata += '{}}}\n'.format(tabs)
        else:
            metadata += '{}End Site\n{}{{\n{}\tOFFSET {:.6f} {:.6f} {:.6f}\n{}}}\n'.format(tabs,
                                                                                           tabs,
                                                                                           tabs, 0, 0, 0, tabs)
            tabs = tabs[:-1]
            metadata += '{}}}\n'.format(tabs)
        if len(hierarchy[joint_parents[joint]]) == 0:
            tabs = tabs[:-1]
        return metadata, tabs

    @staticmethod
    def save_as_bvh(animations, dataset_name=None, subset_name=None, save_file_paths=None,
                    include_default_pose=True, fill=6, frame_time=0.032):
        """
        Saves an animations as a BVH file

        Parameters
        ----------
        :param animations: Dict containing the joint names, offsets, parents, positions, and rotations
            Animation to be saved.

        :param dataset_name: str
            Name of the dataset, e.g., mpi.

        :param subset_name: str
            Name of the subset, e.g., gt, epoch_200.

        :param save_file_paths: str
            Containing directories of the bvh files to be saved.
            If the bvh files exist, they are overwritten.
            If this is None, then the files are saved in numerical order 0, 1, 2, ...

        :param include_default_pose: boolean
            If true, include the default pose at the beginning
        :param fill: int
            Zero padding for file name, if save_file_paths is None. Otherwise, it is not used.

        :param frame_time: float
            Time duration of each frame.
        """

        if not os.path.exists('render'):
            os.makedirs('render')
        dir_name = os.path.join('render', 'bvh')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        dir_name = os.path.join(dir_name, dataset_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if subset_name is not None:
            dir_name = os.path.join(dir_name, subset_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        num_samples = animations['rotations'].shape[0]
        num_frames_max = animations['rotations'].shape[1]
        if 'valid_idx' in animations.keys():
            num_frames = torch.sum(animations['valid_idx'], dim=1).int().detach().cpu().numpy()
        else:
            num_frames = num_frames_max * np.ones(num_samples, dtype=int)
        num_joints = len(animations['joint_parents'])
        save_quats = animations['rotations'].contiguous().view(num_samples, num_frames_max,
                                                               num_joints, -1).detach().cpu().numpy()
        for s in range(num_samples):
            trajectory = animations['positions'][s, :, 0].detach().cpu().numpy()
            save_file_path = os.path.join(
                dir_name, save_file_paths[s] if save_file_paths is not None else str(s).zfill(fill))
            if not os.path.exists(save_file_path):
                os.makedirs(save_file_path)
            save_file_name = os.path.join(save_file_path, 'root.bvh')
            hierarchy = [[] for _ in range(len(animations['joint_parents']))]
            for j in range(len(animations['joint_parents'])):
                if not animations['joint_parents'][j] == -1:
                    hierarchy[animations['joint_parents'][j]].append(j)
            string = ''
            tabs = ''
            joint = 0
            rot_string = 'Xrotation Yrotation Zrotation'
            joint_offsets = animations['joint_offsets'][s].detach().cpu().numpy()
            joint_offsets = np.concatenate((np.zeros_like(joint_offsets[0:1]), joint_offsets), axis=0)
            with open(save_file_name, 'w') as f:
                f.write('{}HIERARCHY\n'.format(tabs))
                f.write('{}ROOT {}\n{{\n'.format(tabs, animations['joint_names'][joint]))
                tabs += '\t'
                f.write('{}OFFSET {:.6f} {:.6f} {:.6f}\n'.format(tabs,
                                                                 joint_offsets[joint][0],
                                                                 joint_offsets[joint][1],
                                                                 joint_offsets[joint][2]))
                f.write('{}CHANNELS 6 Xposition Yposition Zposition {}\n'.format(tabs, rot_string))
                string, tabs = MoCapDataset.traverse_hierarchy(hierarchy, animations['joint_names'],
                                                               joint_offsets, animations['joint_parents'],
                                                               joint, string, tabs, rot_string)
                f.write(string)
                f.write('MOTION\nFrames: {}\nFrame Time: {}\n'.format(num_frames[s] + include_default_pose, frame_time))
                if include_default_pose:
                    string = str(trajectory[0, 0]) + ' ' +\
                        str(trajectory[0, 1]) + ' ' + \
                        str(trajectory[0, 2])
                    for j in range(num_joints * 3):
                        string += ' ' + '{:.6f}'.format(0)
                    f.write(string + '\n')
                for t in range(num_frames[s]):
                    string = str(trajectory[t, 0]) + ' ' + \
                             str(trajectory[t, 1]) + ' ' + \
                             str(trajectory[t, 2])
                    for j in range(num_joints):
                        eulers = np.degrees(Quaternions(save_quats[s, t, j]).euler(order='xyz'))[0]
                        string += ' ' + '{:.6f}'.format(eulers[0]) + \
                                  ' ' + '{:.6f}'.format(eulers[1]) + \
                                  ' ' + '{:.6f}'.format(eulers[2])
                    f.write(string + '\n')
