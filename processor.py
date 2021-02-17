import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from net.ser_att_conv_rnn import AttConvRNN

from torchlight.torchlight.io import IO
from utils import losses

torch.manual_seed(1234)

rec_loss = losses.quat_angle_loss


def find_all_substr(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_epoch_and_loss(path_to_model_files, epoch='best'):
    all_models = os.listdir(path_to_model_files)
    if len(all_models) < 2:
        return '', None, np.inf
    if epoch == 'best':
        loss_list = -1. * np.ones(len(all_models))
        for i, model in enumerate(all_models):
            loss_val = str.split(model, '_')
            if len(loss_val) > 1:
                loss_list[i] = float(loss_val[3])
        if len(loss_list) < 3:
            best_model = all_models[np.argwhere(loss_list == min([n for n in loss_list if n > 0]))[0, 0]]
        else:
            loss_idx = np.argpartition(loss_list, 2)
            best_model = all_models[loss_idx[1]]
        all_underscores = list(find_all_substr(best_model, '_'))
        # return model name, best loss
        return best_model, int(best_model[all_underscores[0] + 1:all_underscores[1]]), \
               float(best_model[all_underscores[2] + 1:all_underscores[3]])
    assert isinstance(epoch, int)
    found_model = None
    for i, model in enumerate(all_models):
        model_epoch = str.split(model, '_')
        if len(model_epoch) > 1 and epoch == int(model_epoch[1]):
            found_model = model
            break
    if found_model is None:
        return '', None, np.inf
    all_underscores = list(find_all_substr(found_model, '_'))
    return found_model, int(found_model[all_underscores[0] + 1:all_underscores[1]]), \
        float(found_model[all_underscores[2] + 1:all_underscores[3]])


class Processor(object):
    """
        Processor for emotive gesture generation
    """

    def __init__(self, args, data_path, data_loader, C, H, W, D,
                 min_train_epochs=20,
                 zfill=6,
                 save_path=None):

        self.args = args
        self.dataset = args.dataset_ser
        self.channel_map = {
            'Xrotation': 'x',
            'Yrotation': 'y',
            'Zrotation': 'z'
        }
        self.data_loader = data_loader
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.io = IO(
            self.args.work_dir_ser,
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # model
        self.C = C
        self.H = H
        self.W = W
        self.D = D

        self.L1 = 128
        self.L2 = 256
        self.gru_cell_units = 128
        self.attention_size = 1
        self.num_linear = 768
        self.pool_stride_height = 2
        self.pool_stride_width = 4
        self.F1 = 64
        self.bidirectional = True
        self.dropout_keep_prob = 1.

        self.pred_loss_func = nn.SmoothL1Loss()
        self.best_loss = np.inf
        self.loss_updated = False
        self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_loss_epoch = None
        self.min_train_epochs = min_train_epochs
        self.zfill = zfill
        self.ser_model = AttConvRNN(C=self.C, H=self.H, W=self.W, D=self.D,
                                    L1=self.L1, L2=self.L2, gru_cell_units=self.gru_cell_units,
                                    attention_size=self.attention_size, num_linear=self.num_linear,
                                    pool_stride_height=self.pool_stride_height,
                                    pool_stride_width=self.pool_stride_width,
                                    F1=self.F1, bidirectional=self.bidirectional,
                                    dropout_keep_prob=self.dropout_keep_prob)
        if self.args.use_multiple_gpus and torch.cuda.device_count() > 1:
            self.args.batch_size *= torch.cuda.device_count()
            self.ser_model = nn.DataParallel(self.ser_model)
        self.ser_model.to(torch.cuda.current_device())
        print('Total training data:\t\t{}'.format(len(self.data_loader['train_data'])))
        print('Total evaluation data:\t\t{}'.format(len(self.data_loader['eval_data'])))
        print('Total testing data:\t\t\t{}'.format(len(self.data_loader['test_data'])))
        print('Training with batch size:\t{}'.format(self.args.batch_size))

        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.ser_model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.ser_model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr
        self.tf = self.args.base_tr

    def process_data(self, data, poses, quat, trans, affs):
        data = data.float().cuda()
        poses = poses.float().cuda()
        quat = quat.float().cuda()
        trans = trans.float().cuda()
        affs = affs.float().cuda()
        return data, poses, quat, trans, affs

    def load_model_at_epoch(self, epoch='best'):
        model_name, self.best_loss_epoch, self.best_loss = \
            get_epoch_and_loss(self.args.work_dir_ser, epoch=epoch)
        model_found = False
        try:
            loaded_vars = torch.load(os.path.join(self.args.work_dir_ser, model_name))
            self.ser_model.load_state_dict(loaded_vars['model_dict'])
            model_found = True
        except (FileNotFoundError, IsADirectoryError):
            if epoch == 'best':
                print('Warning! No saved model found.')
            else:
                print('Warning! No saved model found at epoch {:d}.'.format(epoch))
        return model_found

    def adjust_lr(self):
        self.lr = self.lr * self.args.lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def adjust_tf(self):
        if self.meta_info['epoch'] > 20:
            self.tf = self.tf * self.args.tf_decay

    def show_epoch_info(self):

        print_epochs = [self.best_loss_epoch if self.best_loss_epoch is not None else 0]
        best_metrics = [self.best_loss]
        i = 0
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}. Best so far: {} (epoch: {:d}).'.
                              format(k, v, best_metrics[i], print_epochs[i]))
            i += 1
        if self.args.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.args.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def count_parameters(self):
        return sum(p.numel() for p in self.ser_model.parameters() if p.requires_grad)

    def yield_batch(self, train):
        batch_data = torch.zeros((self.args.batch_size, self.C, self.H, self.W)).cuda()
        batch_labels = torch.zeros((self.args.batch_size, self.D)).cuda()

        if train:
            data_np = self.data_loader['train_data']
            labels_np = self.data_loader['train_labels']
        else:
            data_np = self.data_loader['eval_data']
            labels_np = self.data_loader['eval_labels']

        num_data = len(data_np)
        pseudo_passes = (num_data + self.args.batch_size - 1) // self.args.batch_size
        prob_dist = np.ones(num_data) / float(num_data)

        for p in range(pseudo_passes):
            rand_keys = np.random.choice(num_data, size=self.args.batch_size, replace=True, p=prob_dist)
            for i, k in enumerate(rand_keys):
                batch_data[i] = torch.from_numpy(data_np[k])
                batch_labels[i] = torch.from_numpy(labels_np[k])

            yield batch_data, batch_labels

    def return_batch(self, batch_size, dataset, randomized=True):
        data_np = dataset['data']
        labels_np = dataset['labels']
        if len(batch_size) > 1:
            rand_keys = np.copy(batch_size)
            batch_size = len(batch_size)
        else:
            batch_size = batch_size[0]
            num_data = len(data_np)
            prob_dist = np.ones(num_data) / float(num_data)
            if randomized:
                rand_keys = np.random.choice(num_data, size=batch_size, replace=False, p=prob_dist)
            else:
                rand_keys = np.arange(batch_size)

        batch_data = torch.zeros((batch_size, self.C, self.H, self.W)).cuda()
        batch_labels = torch.zeros((batch_size, self.D)).cuda()

        for i, k in enumerate(rand_keys):
            batch_data[i] = torch.from_numpy(data_np[k])
            batch_labels[i] = torch.from_numpy(labels_np[k])

        return batch_data, batch_labels

    def forward_pass(self, data, labels_gt):
        self.optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            labels_pred = self.ser_model(data)
            # labels_pred_np = labels_pred.detach().cpu().numpy()
            # labels_gt_np = labels_gt.detach().cpu().numpy()
            total_loss = 10. * self.pred_loss_func(labels_pred, labels_gt)
        return total_loss

    def per_train(self):

        self.ser_model.train()
        batch_loss = 0.
        num_batches = 0.

        for train_data_wav, train_labels in self.yield_batch(train=True):
            train_loss = self.forward_pass(train_data_wav, train_labels)
            train_loss.backward()
            # nn.utils.clip_grad_norm_(self.ser_model.parameters(), self.args.gradient_clip)
            self.optimizer.step()

            # Compute statistics
            batch_loss += train_loss.item()
            num_batches += 1

            # statistics
            self.iter_info['loss'] = train_loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.iter_info['tf'] = '{:.6f}'.format(self.tf)
            self.show_iter_info()
            self.meta_info['iter'] += 1

        batch_loss /= num_batches
        self.epoch_info['mean_loss'] = batch_loss
        self.show_epoch_info()
        self.io.print_timer()
        self.adjust_lr()
        self.adjust_tf()

    def per_eval(self):

        self.ser_model.eval()
        eval_loader = self.data_loader['eval_data']
        batch_loss = 0.
        num_batches = 0.

        for eval_data_wav, eval_labels in self.yield_batch(train=False):
            with torch.no_grad():
                eval_loss = self.forward_pass(eval_data_wav, eval_labels)
                batch_loss += eval_loss.item()
                num_batches += 1

        batch_loss /= num_batches
        self.epoch_info['mean_loss'] = batch_loss
        if self.epoch_info['mean_loss'] < self.best_loss and self.meta_info['epoch'] > self.min_train_epochs:
            self.best_loss = self.epoch_info['mean_loss']
            self.best_loss_epoch = self.meta_info['epoch']
            self.loss_updated = True
        else:
            self.loss_updated = False
        self.show_epoch_info()

    def train(self):

        if self.args.load_last_best:
            model_found = self.load_model_at_epoch(epoch=self.args.start_epoch)
            if not model_found and self.args.start_epoch is not 'best':
                print('Warning! Trying to load best known model: '.format(self.args.start_epoch), end='')
                model_found = self.load_model_at_epoch(epoch='best')
                self.args.start_epoch = self.best_loss_epoch if model_found else 0
                print('loaded.')
                if not model_found:
                    print('Warning! Starting at epoch 0')
                    self.args.start_epoch = 0
        else:
            self.args.start_epoch = 0
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            self.io.print_log('Done.')

            # evaluation
            if (epoch % self.args.eval_interval == 0) or (
                    epoch + 1 == self.args.num_epoch):
                self.io.print_log('Eval epoch: {}'.format(epoch))
                self.per_eval()
                self.io.print_log('Done.')

            # save model and weights
            if self.loss_updated or epoch % self.args.save_interval == 0:
                torch.save({'model_dict': self.ser_model.state_dict()},
                           os.path.join(self.args.work_dir_ser, 'epoch_{}_loss_{:.4f}_model.pth.tar'.
                                        format(epoch, self.epoch_info['mean_loss'])))

    def copy_prefix(self, var, prefix_length=None):
        if prefix_length is None:
            prefix_length = self.prefix_length
        return [var[s, :prefix_length].unsqueeze(0) for s in range(var.shape[0])]

    def generate_motion(self, load_saved_model=True, samples_to_generate=10, randomized=True, epoch='best'):

        if load_saved_model:
            self.load_model_at_epoch(epoch=epoch)
        self.ser_model.eval()
        test_loader = self.data_loader['test']

        start_time = time.time()
        joint_offsets, pos, affs, quat, quat_eval_idx, \
        text, text_eval_idx, perceived_emotion, perceived_polarity, \
        acting_task, gender, age, handedness, \
        native_tongue = self.return_batch([samples_to_generate], test_loader, randomized=randomized)
        with torch.no_grad():
            joint_lengths = torch.norm(joint_offsets, dim=-1)
            scales, _ = torch.max(joint_lengths, dim=-1)
            quat_pred = torch.zeros_like(quat)
            quat_pred[:, 0] = torch.cat(quat_pred.shape[0] * [self.quats_sos]).view(quat_pred[:, 0].shape)

            quat_pred, quat_pred_pre_norm = self.ser_model(text, perceived_emotion, perceived_polarity,
                                                       acting_task, gender, age, handedness, native_tongue,
                                                       quat[:, :-1], joint_lengths / scales[..., None])
            # text_latent = self.ser_model(text, intended_emotion, intended_polarity,
            #                          acting_task, gender, age, handedness, native_tongue, only_encoder=True)
            # for t in range(1, self.T):
            #     quat_pred_curr, _ = self.ser_model(text_latent, quat=quat_pred[:, 0:t],
            #                                    offset_lengths=joint_lengths / scales[..., None],
            #                                    only_decoder=True)
            #     quat_pred[:, t:t + 1] = quat_pred_curr[:, -1:].clone()

            # for s in range(len(quat_pred)):
            #     quat_pred[s] = qfix(quat_pred[s].view(quat_pred[s].shape[0],
            #                                           self.V, -1)).view(quat_pred[s].shape[0], -1)
            quat_pred = torch.cat((quat[:, 1:2], quat_pred), dim=1)
            quat_pred = qfix(quat_pred.view(quat_pred.shape[0], quat_pred.shape[1],
                                            self.V, -1)).view(quat_pred.shape[0], quat_pred.shape[1], -1)
            quat_pred = quat_pred[:, 1:]

            quat_np = quat.detach().cpu().numpy()
            quat_pred_np = quat_pred.detach().cpu().numpy()
            root_pos = torch.zeros(quat_pred.shape[0], quat_pred.shape[1], self.C).cuda()
            pos_pred = MocapDataset.forward_kinematics(quat_pred.contiguous().view(
                quat_pred.shape[0], quat_pred.shape[1], -1, self.D), root_pos, self.joint_parents,
                torch.cat((root_pos[:, 0:1], joint_offsets), dim=1).unsqueeze(1))

        animation_pred = {
            'joint_names': self.joint_names,
            'joint_offsets': joint_offsets,
            'joint_parents': self.joint_parents,
            'positions': pos_pred,
            'rotations': quat_pred,
            'eval_idx': quat_eval_idx
        }
        MocapDataset.save_as_bvh(animation_pred,
                                 dataset_name=self.dataset,
                                 subset_name='test_epoch_{}'.format(epoch),
                                 include_default_pose=False)
        end_time = time.time()
        print('Time taken: {} secs.'.format(end_time - start_time))
        shifted_pos = pos - pos[:, :, 0:1]
        animation = {
            'joint_names': self.joint_names,
            'joint_offsets': joint_offsets,
            'joint_parents': self.joint_parents,
            'positions': shifted_pos,
            'rotations': quat,
            'eval_idx': quat_eval_idx
        }

        MocapDataset.save_as_bvh(animation,
                                 dataset_name=self.dataset,
                                 subset_name='gt',
                                 include_default_pose=False)
        pos_pred_np = pos_pred.contiguous().view(pos_pred.shape[0],
                                                 pos_pred.shape[1], -1).permute(0, 2, 1). \
            detach().cpu().numpy()
        display_animations(pos_pred_np, self.joint_parents, save=True,
                           dataset_name=self.dataset,
                           subset_name='epoch_' + str(self.best_loss_epoch),
                           overwrite=True)
