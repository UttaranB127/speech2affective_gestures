import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pyarrow
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as j
from torchlight.torchlight.io import IO

from net.ser_att_conv_rnn import AttConvRNN
from net.multimodal_context_net import PoseGenerator, AffDiscriminator
from utils.gen_utils import create_video_and_save
from utils import losses
from utils.ted_db_utils import convert_pose_seq_to_dir_vec, make_audio_fixed_length


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


def get_epoch_and_loss(path_to_model_files, phase, emo_as_cats, epoch='best'):
    all_models = os.listdir(path_to_model_files)
    if len(all_models) < 2:
        if phase == 'ser':
            return '', None, 0. if emo_as_cats else -np.inf, np.inf
        if phase == 's2eg':
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
        if phase == 'ser':
            return best_model, int(best_model[all_underscores[0] + 1:all_underscores[1]]),\
                float(best_model[all_underscores[2] + 1:all_underscores[3]]), \
                float(best_model[all_underscores[4] + 1:all_underscores[5]])
        if phase == 's2eg':
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
        if phase == 'ser':
            return '', None, 0. if emo_as_cats else -np.inf, np.inf
        if phase == 'se2g':
            return '', None, np.inf
    all_underscores = list(find_all_substr(found_model, '_'))
    if phase == 'ser':
        return found_model, int(found_model[all_underscores[0] + 1:all_underscores[1]]),\
            float(found_model[all_underscores[2] + 1:all_underscores[3]]),\
            float(found_model[all_underscores[4] + 1:all_underscores[5]])
    if phase == 's2eg':
        return found_model, int(found_model[all_underscores[0] + 1:all_underscores[1]]),\
            float(found_model[all_underscores[2] + 1:all_underscores[3]])


class Processor(object):
    """
        Processor for emotive gesture generation
    """

    def __init__(self, args, config_args, data_path, data_loader,
                 C, H, W, EC, ED, P, T,
                 min_train_epochs=20,
                 zfill=6,
                 save_path=None):

        self.args = args
        self.config_args = config_args
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
        self.EC = EC
        self.ED = ED
        self.P = P
        self.T = T
        self.num_labels = self.EC if self.args.emo_as_cats else self.ED

        self.L1 = 128
        self.L2 = 256
        self.L3 = 256
        self.L4 = 256
        self.gru_cell_units = 128
        self.attention_size = 5
        self.pool_stride_height = 2
        self.pool_stride_width = 4
        self.F1 = 768
        self.F2 = 64
        self.bidirectional = True
        self.dropout_prob = 0.

        self.pred_loss_func = nn.CrossEntropyLoss() if self.args.emo_as_cats else nn.L1Loss()
        self.best_ser_accu = 0. if self.args.emo_as_cats else -np.inf
        self.ser_accu_updated = False
        self.ser_step_epochs = [math.ceil(float(self.args.ser_num_epoch * x)) for x in self.args.step]
        self.best_ser_accu_epoch = None
        self.best_ser_accu_loss = None
        self.best_s2eg_loss = np.inf
        self.best_s2eg_loss_epoch = None
        self.s2eg_loss_updated = False
        self.min_train_epochs = min_train_epochs
        self.zfill = zfill
        self.ser_model = AttConvRNN(C=self.C, H=self.H, W=self.W, EC=self.num_labels,
                                    L1=self.L1, L2=self.L2, gru_cell_units=self.gru_cell_units,
                                    attention_size=self.attention_size, F1=self.F1,
                                    pool_stride_height=self.pool_stride_height,
                                    pool_stride_width=self.pool_stride_width,
                                    F2=self.F2, bidirectional=self.bidirectional,
                                    dropout_prob=self.dropout_prob)
        if not self.args.train_ser:
            lang_model = self.data_loader['train_data_s2eg'].lang_model
            self.train_speaker_model = self.data_loader['train_data_s2eg'].speaker_model
            self.eval_speaker_model = self.data_loader['eval_data_s2eg'].speaker_model
            self.test_speaker_model = self.data_loader['test_data_s2eg'].speaker_model
            self.s2eg_generator = PoseGenerator(self.config_args,
                                                n_words=lang_model.n_words,
                                                word_embed_size=self.config_args.wordembed_dim,
                                                word_embeddings=lang_model.word_embedding_weights,
                                                labels_size=self.num_labels,
                                                z_obj=self.train_speaker_model,
                                                pose_dim=self.P)
            self.s2eg_discriminator = AffDiscriminator(self.P, self.EC if self.args.emo_as_cats else self.ED)
        else:
            lang_model, self.train_speaker_model,\
                self.eval_speaker_model, self.test_speaker_model,\
                self.s2eg_generator, self.s2eg_discriminator = [None] * 6

        if self.args.use_multiple_gpus and torch.cuda.device_count() > 1:
            self.args.batch_size *= torch.cuda.device_count()
            self.ser_model = nn.DataParallel(self.ser_model)
            if not self.args.train_ser:
                self.s2eg_generator = nn.DataParallel(self.s2eg_generator)
                self.s2eg_discriminator = nn.DataParallel(self.s2eg_discriminator)
        self.ser_model.to(torch.cuda.current_device())
        if not self.args.train_ser:
            self.s2eg_generator.to(torch.cuda.current_device())
            self.s2eg_discriminator.to(torch.cuda.current_device())
        self.conv2_weights = []

        if self.args.train_ser:
            print('Total ser training data:\t\t{}'.format(len(self.data_loader['train_data_ser'])))
            print('Total ser evaluation data:\t\t{}'.format(len(self.data_loader['eval_data_ser'])))
            print('Total ser testing data:\t\t\t{}'.format(len(self.data_loader['test_data_ser'])))
            print('Training ser with batch size:\t{}'.format(self.args.batch_size))

        if self.args.train_s2eg:
            print('Total s2eg training data:\t\t{}'.format(len(self.data_loader['train_data_s2eg_wav'])))
            print('Total s2eg evaluation data:\t\t{}'.format(len(self.data_loader['eval_data_s2eg_wav'])))
            print('Total s2eg testing data:\t\t\t{}'.format(len(self.data_loader['test_data_s2eg_wav'])))
            print('Training s2eg with batch size:\t{}'.format(self.args.batch_size))

        # ser optimizer
        if self.args.ser_optimizer == 'SGD':
            self.ser_optimizer = optim.SGD(
                self.ser_model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.ser_optimizer == 'Adam':
            self.ser_optimizer = optim.Adam(
                self.ser_model.parameters(),
                lr=self.args.base_lr_ser,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr_ser = self.args.base_lr_ser
        self.lr_s2eg_gen = self.config_args.learning_rate
        self.lr_s2eg_dis = self.config_args.learning_rate * self.config_args.discriminator_lr_weight

        # s2eg optimizers
        if not self.args.train_ser:
            self.s2eg_gen_optimizer = optim.Adam(self.s2eg_generator.parameters(),
                                                 lr=self.lr_s2eg_gen, betas=(0.5, 0.999))
            self.s2eg_dis_optimizer = torch.optim.Adam(
                self.s2eg_discriminator.parameters(),
                lr=self.lr_s2eg_dis,
                betas=(0.5, 0.999))

    def process_data(self, data, poses, quat, trans, affs):
        data = data.float().cuda()
        poses = poses.float().cuda()
        quat = quat.float().cuda()
        trans = trans.float().cuda()
        affs = affs.float().cuda()
        return data, poses, quat, trans, affs

    def load_model_at_epoch(self, phase, epoch='best'):
        work_dir = self.args.work_dir_ser if phase == 'ser'\
            else (self.args.work_dir_s2eg if phase == 's2eg' else None)
        model_name = None
        if phase == 'ser':
            model_name, self.best_ser_accu_epoch, \
                self.best_ser_accu, self.best_ser_accu_loss =\
                get_epoch_and_loss(work_dir, 'ser', emo_as_cats=self.args.emo_as_cats, epoch=epoch)
        elif phase == 's2eg':
            model_name, self.best_s2eg_loss_epoch, self.best_s2eg_loss =\
                get_epoch_and_loss(work_dir, 's2eg', emo_as_cats=self.args.emo_as_cats, epoch=epoch)
        model_found = False
        try:
            loaded_vars = torch.load(j(work_dir, model_name))
            if phase == 'ser':
                self.ser_model.load_state_dict(loaded_vars['ser_model_dict'])
            elif phase == 's2eg':
                self.s2eg_generator.load_state_dict(loaded_vars['gen_model_dict'])
                self.s2eg_discriminator.load_state_dict(loaded_vars['dis_model_dict'])
            model_found = True
        except (FileNotFoundError, IsADirectoryError):
            if epoch == 'best':
                print('Warning! No saved model found.')
            else:
                print('Warning! No saved model found at epoch {:d}.'.format(epoch))
        return model_found

    def adjust_lr_ser(self):
        self.lr_ser = self.lr_ser * self.args.lr_ser_decay
        for param_group in self.ser_optimizer.param_groups:
            param_group['lr'] = self.lr_ser

    def adjust_lr_s2eg(self):
        self.lr_s2eg_gen = self.lr_s2eg_gen * self.args.lr_s2eg_decay
        for param_group in self.s2eg_gen_optimizer.param_groups:
            param_group['lr'] = self.lr_s2eg_gen

        self.lr_s2eg_dis = self.lr_s2eg_dis * self.args.lr_s2eg_decay
        for param_group in self.s2eg_dis_optimizer.param_groups:
            param_group['lr'] = self.lr_s2eg_dis

    def show_epoch_info(self):

        best_metrics = []
        print_epochs = []
        if self.args.train_ser:
            best_metrics = [self.best_ser_accu, self.best_ser_accu_loss]
            print_epochs = [self.best_ser_accu_epoch
                            if self.best_ser_accu_epoch is not None else 0] * len(best_metrics)
        if self.args.train_s2eg:
            best_metrics = [self.best_s2eg_loss]
            print_epochs = [self.best_s2eg_loss_epoch
                            if self.best_s2eg_loss_epoch is not None else 0] * len(best_metrics)
        i = 0
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}. Best so far: {:.4f} (epoch: {:d}).'.
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
        batch_data_ser = torch.zeros((self.args.batch_size, self.C, self.H, self.W)).cuda()
        batch_data_s2eg = torch.zeros((self.args.batch_size, self.C, self.H, self.W)).cuda()
        batch_labels_cat = torch.zeros(self.args.batch_size).long().cuda()
        batch_labels_dim = torch.zeros((self.args.batch_size, self.ED)).float().cuda()
        batch_word_seq_tensor = torch.zeros((self.args.batch_size, self.T)).long().cuda()
        batch_word_seq_lengths = torch.zeros(self.args.batch_size).long().cuda()
        batch_extended_word_seq = torch.zeros((self.args.batch_size, self.T)).long().cuda()
        batch_pose_seq = torch.zeros((self.args.batch_size, self.T, self.P + self.C)).float().cuda()
        batch_vec_seq = torch.zeros((self.args.batch_size, self.T, self.P)).float().cuda()
        batch_audio = torch.zeros((self.args.batch_size, 36267)).float().cuda()
        batch_spectrogram = torch.zeros((self.args.batch_size, 128, 70)).float().cuda()
        batch_vid_indices = torch.zeros(self.args.batch_size).long().cuda()

        if train:
            data_ser_np = self.data_loader['train_data_ser']
            data_s2eg_np = self.data_loader['train_data_s2eg_wav']
            data_s2eg = self.data_loader['train_data_s2eg']
            labels_cat_np = self.data_loader['train_labels_cat']
            labels_dim_np = self.data_loader['train_labels_dim']
        else:
            data_ser_np = self.data_loader['eval_data_ser']
            data_s2eg_np = self.data_loader['eval_data_s2eg_wav']
            data_s2eg = self.data_loader['eval_data_s2eg']
            labels_cat_np = self.data_loader['eval_labels_cat']
            labels_dim_np = self.data_loader['eval_labels_dim']

        num_data = len(data_ser_np)
        pseudo_passes = (num_data + self.args.batch_size - 1) // self.args.batch_size
        prob_dist = np.ones(num_data) / float(num_data)

        def extend_word_seq(lang, words, end_time=None):
            n_frames = data_s2eg.n_poses
            if end_time is None:
                end_time = aux_info['end_time']
            frame_duration = (end_time - aux_info['start_time']) / n_frames

            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            if data_s2eg.remove_word_timing:
                n_words = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        n_words += 1
                space = int(n_frames / (n_words + 1))
                for word_idx in range(n_words):
                    idx = (word_idx + 1) * space
                    extended_word_indices[idx] = lang.get_word_index(words[word_idx][0])
            else:
                prev_idx = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        extended_word_indices[idx] = lang.get_word_index(word[0])
                        # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
                        prev_idx = idx
            return torch.Tensor(extended_word_indices).long()

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        for p in range(pseudo_passes):
            rand_keys = np.random.choice(num_data, size=self.args.batch_size, replace=True, p=prob_dist)
            for i, k in enumerate(rand_keys):
                batch_data_ser[i] = torch.from_numpy(data_ser_np[k])
                batch_labels_cat[i] = torch.from_numpy(np.where(labels_cat_np[k])[0])
                batch_labels_dim[i] = torch.from_numpy(labels_dim_np[k])

                if self.args.train_s2eg:
                    with data_s2eg.lmdb_env.begin(write=False) as txn:
                        key = '{:010}'.format(k).encode('ascii')
                        sample = txn.get(key)
                        sample = pyarrow.deserialize(sample)
                        word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample

                        # vid_name = sample[-1]['vid']
                        # clip_start = str(sample[-1]['start_time'])
                        # clip_end = str(sample[-1]['end_time'])
                        batch_data_s2eg[i] = torch.from_numpy(data_s2eg_np[k])

                    duration = aux_info['end_time'] - aux_info['start_time']
                    do_clipping = True

                    if do_clipping:
                        sample_end_time = aux_info['start_time'] + duration * data_s2eg.n_poses / vec_seq.shape[0]
                        audio = make_audio_fixed_length(audio, data_s2eg.expected_audio_length)
                        spectrogram = spectrogram[:, 0:data_s2eg.expected_spectrogram_length]
                        vec_seq = vec_seq[0:data_s2eg.n_poses]
                        pose_seq = pose_seq[0:data_s2eg.n_poses]
                    else:
                        sample_end_time = None

                    # to tensors
                    word_seq_tensor = words_to_tensor(data_s2eg.lang_model, word_seq, sample_end_time)
                    extended_word_seq = extend_word_seq(data_s2eg.lang_model, word_seq, sample_end_time)
                    vec_seq = torch.from_numpy(vec_seq).reshape((vec_seq.shape[0], -1)).float()
                    pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
                    audio = torch.from_numpy(audio).float()
                    spectrogram = torch.from_numpy(spectrogram)

                    batch_word_seq_tensor[i, :len(word_seq_tensor)] = word_seq_tensor
                    batch_word_seq_lengths[i] = len(word_seq_tensor)
                    batch_extended_word_seq[i] = extended_word_seq
                    batch_pose_seq[i] = pose_seq
                    batch_vec_seq[i] = vec_seq
                    batch_audio[i] = audio
                    batch_spectrogram[i] = spectrogram
                    # speaker input
                    if train:
                        if self.train_speaker_model and self.train_speaker_model.__class__.__name__ == 'Vocab':
                            batch_vid_indices[i] =\
                                torch.LongTensor([self.train_speaker_model.word2index[aux_info['vid']]])
                    else:
                        if self.eval_speaker_model and self.eval_speaker_model.__class__.__name__ == 'Vocab':
                            batch_vid_indices[i] =\
                                torch.LongTensor([self.eval_speaker_model.word2index[aux_info['vid']]])

            yield batch_data_ser, batch_labels_cat, batch_labels_dim,\
                batch_word_seq_tensor, batch_word_seq_lengths, batch_extended_word_seq,\
                batch_pose_seq, batch_vec_seq, batch_audio, batch_spectrogram, batch_vid_indices

    def return_batch(self, batch_size, randomized=True):

        data_ser_np = self.data_loader['test_data_ser']
        data_s2eg_np = self.data_loader['test_data_s2eg_wav']
        data_s2eg = self.data_loader['test_data_s2eg']
        labels_cat_np = self.data_loader['test_labels_cat']
        labels_dim_np = self.data_loader['test_labels_dim']

        if len(batch_size) > 1:
            rand_keys = np.copy(batch_size)
            batch_size = len(batch_size)
        else:
            batch_size = batch_size[0]
            num_data = len(data_ser_np)
            prob_dist = np.ones(num_data) / float(num_data)
            if randomized:
                rand_keys = np.random.choice(num_data, size=batch_size, replace=False, p=prob_dist)
            else:
                rand_keys = np.arange(batch_size)

        batch_data_ser = torch.zeros((batch_size, self.C, self.H, self.W)).cuda()
        batch_data_s2eg = torch.zeros((batch_size, self.C, self.H, self.W)).cuda()
        batch_labels_cat = torch.zeros(batch_size).long().cuda()
        batch_labels_dim = torch.zeros((batch_size, self.ED)).float().cuda()
        batch_words = [[] for _ in range(batch_size)]
        batch_aux_info = [[] for _ in range(batch_size)]
        batch_word_seq_tensor = torch.zeros((batch_size, self.T)).long().cuda()
        batch_word_seq_lengths = torch.zeros(batch_size).long().cuda()
        batch_extended_word_seq = torch.zeros((batch_size, self.T)).long().cuda()
        batch_pose_seq = torch.zeros((batch_size, self.T, self.P + self.C)).float().cuda()
        batch_vec_seq = torch.zeros((batch_size, self.T, self.P)).float().cuda()
        batch_target_seq = torch.zeros((batch_size, self.T, self.P)).float().cuda()
        batch_audio = torch.zeros((batch_size, 36267)).float().cuda()
        batch_spectrogram = torch.zeros((batch_size, 128, 70)).float().cuda()
        batch_vid_indices = torch.zeros(batch_size).long().cuda()

        def extend_word_seq(lang, words, end_time=None):
            n_frames = data_s2eg.n_poses
            if end_time is None:
                end_time = aux_info['end_time']
            frame_duration = (end_time - aux_info['start_time']) / n_frames

            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            if data_s2eg.remove_word_timing:
                n_words = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        n_words += 1
                space = int(n_frames / (n_words + 1))
                for word_idx in range(n_words):
                    idx = (word_idx + 1) * space
                    extended_word_indices[idx] = lang.get_word_index(words[word_idx][0])
            else:
                prev_idx = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        extended_word_indices[idx] = lang.get_word_index(word[0])
                        # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
                        prev_idx = idx
            return torch.Tensor(extended_word_indices).long()

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        for i, k in enumerate(rand_keys):
            batch_data_ser[i] = torch.from_numpy(data_ser_np[k])
            batch_labels_cat[i] = torch.from_numpy(np.where(labels_cat_np[k])[0])
            batch_labels_dim[i] = torch.from_numpy(labels_dim_np[k])

            if not self.args.train_ser:
                with data_s2eg.lmdb_env.begin(write=False) as txn:
                    key = '{:010}'.format(k).encode('ascii')
                    sample = txn.get(key)
                    sample = pyarrow.deserialize(sample)
                    word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample

                    batch_data_s2eg[i] = torch.from_numpy(data_s2eg_np[k])
                    # for selected_vi in range(len(word_seq)):  # make start time of input text zero
                    #     word_seq[selected_vi][1] -= aux_info['start_time']  # start time
                    #     word_seq[selected_vi][2] -= aux_info['start_time']  # end time
                    batch_words[i] = [word_seq[i][0] for i in range(len(word_seq))]
                    batch_aux_info[i] = aux_info

                duration = aux_info['end_time'] - aux_info['start_time']
                do_clipping = True

                if do_clipping:
                    sample_end_time = aux_info['start_time'] + duration * data_s2eg.n_poses / vec_seq.shape[0]
                    audio = make_audio_fixed_length(audio, data_s2eg.expected_audio_length)
                    spectrogram = spectrogram[:, 0:data_s2eg.expected_spectrogram_length]
                    vec_seq = vec_seq[0:data_s2eg.n_poses]
                    pose_seq = pose_seq[0:data_s2eg.n_poses]
                else:
                    sample_end_time = None

                # to tensors
                word_seq_tensor = words_to_tensor(data_s2eg.lang_model, word_seq, sample_end_time)
                extended_word_seq = extend_word_seq(data_s2eg.lang_model, word_seq, sample_end_time)
                vec_seq = torch.from_numpy(vec_seq).reshape((vec_seq.shape[0], -1)).float()
                pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
                target_seq = convert_pose_seq_to_dir_vec(pose_seq)
                target_seq = target_seq.reshape(target_seq.shape[0], -1)
                target_seq -= np.reshape(self.data_loader['test_data_s2eg'].mean_dir_vec, -1)
                audio = torch.from_numpy(audio).float()
                spectrogram = torch.from_numpy(spectrogram)

                batch_word_seq_tensor[i, :len(word_seq_tensor)] = word_seq_tensor
                batch_word_seq_lengths[i] = len(word_seq_tensor)
                batch_extended_word_seq[i] = extended_word_seq
                batch_pose_seq[i] = pose_seq
                batch_vec_seq[i] = vec_seq
                batch_target_seq[i] = torch.from_numpy(target_seq).float()
                batch_audio[i] = audio
                batch_spectrogram[i] = spectrogram
                # speaker input
                if self.test_speaker_model and self.test_speaker_model.__class__.__name__ == 'Vocab':
                    batch_vid_indices[i] =\
                        torch.LongTensor([self.test_speaker_model.word2index[aux_info['vid']]])

        return batch_data_ser, batch_labels_cat, batch_labels_dim, batch_words,\
            batch_aux_info, batch_word_seq_tensor, batch_word_seq_lengths, batch_extended_word_seq,\
            batch_pose_seq, batch_vec_seq, batch_target_seq, batch_audio, batch_spectrogram, batch_vid_indices

    def forward_pass_ser(self, data, labels_gt=None):
        self.ser_optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            labels_pred_raw = self.ser_model(data)
            # labels_pred_np = labels_pred.detach().cpu().numpy()
            # labels_gt_np = labels_gt.detach().cpu().numpy()
            if self.args.emo_as_cats:
                labels_pred = labels_pred_raw
            else:
                # labels_pred = torch.sigmoid(labels_pred_raw)
                labels_pred = labels_pred_raw
                labels_pred_diff = labels_pred[1:] - labels_pred[:-1]
            # total_loss = None if labels_gt is None else self.pred_loss_func(labels_pred, labels_gt)
            total_loss = None if labels_gt is None else ((self.pred_loss_func(labels_pred, labels_gt) +
                                                         (0. if self.args.emo_as_cats else
                                                          self.pred_loss_func(labels_pred_diff,
                                                                              labels_gt[1:] - labels_gt[:-1]))) * 1.)
            max_idx = torch.argmax(labels_pred, -1, keepdim=True)
            labels_one_hot = torch.FloatTensor(labels_pred.shape).cuda()
            labels_one_hot.zero_()
            labels_one_hot.scatter_(1, max_idx, 1)
        return total_loss, torch.argmax(labels_pred, dim=-1) if self.args.emo_as_cats else labels_pred, labels_one_hot

    @staticmethod
    def add_noise(data):
        noise = torch.randn_like(data) * 0.1
        return data + noise

    def forward_pass_s2eg(self, in_text, in_audio, in_emo_labels, target_poses, vid_indices, train,
                          target_seq=None, words=None, aux_info=None, save_path=None, make_video=False):
        warm_up_epochs = self.config_args.loss_warmup
        use_noisy_target = False

        # make pre seq input
        pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
        pre_seq[:, 0:self.config_args.n_pre_poses, :-1] = target_poses[:, 0:self.config_args.n_pre_poses]
        pre_seq[:, 0:self.config_args.n_pre_poses, -1] = 1  # indicating bit for constraints

        ###########################################################################################
        # train D
        dis_error = None
        if self.meta_info['epoch'] > warm_up_epochs and self.config_args.loss_gan_weight > 0.0:
            self.s2eg_dis_optimizer.zero_grad()

            # out shape (batch x seq x dim)
            out_dir_vec, *_ = self.s2eg_generator(pre_seq, in_text, in_audio, in_emo_labels, vid_indices)

            if use_noisy_target:
                noise_target = Processor.add_noise(target_poses)
                noise_out = Processor.add_noise(out_dir_vec.detach())
                dis_real = self.s2eg_discriminator(noise_target, in_emo_labels, in_text)
                dis_fake = self.s2eg_discriminator(noise_out, in_emo_labels, in_text)
            else:
                dis_real = self.s2eg_discriminator(target_poses, in_emo_labels, in_text)
                dis_fake = self.s2eg_discriminator(out_dir_vec.detach(), in_emo_labels, in_text)

            dis_error = torch.sum(-torch.mean(torch.log(dis_real + 1e-8) + torch.log(1 - dis_fake + 1e-8)))  # ns-gan
            if train:
                dis_error.backward()
                self.s2eg_dis_optimizer.step()

        ###########################################################################################
        # train G
        self.s2eg_gen_optimizer.zero_grad()

        # decoding
        out_dir_vec, z, z_mu, z_log_var = self.s2eg_generator(pre_seq, in_text, in_audio, in_emo_labels, vid_indices)

        # make a video
        assert not make_video or (make_video and target_seq is not None), \
            'target_seq cannot be None when make_video is True'
        assert not make_video or (make_video and words is not None), \
            'words cannot be None when make_video is True'
        assert not make_video or (make_video and aux_info is not None), \
            'aux_info cannot be None when make_video is True'
        assert not make_video or (make_video and save_path is not None), \
            'save_path cannot be None when make_video is True'
        if make_video:
            sentence_words = []
            for word in words:
                sentence_words.append(word)
            sentences = [' '.join(sentence_word) for sentence_word in sentence_words]

            for vid_idx in range(len(aux_info)):
                filename_prefix = '{}_{}'.format(aux_info[vid_idx]['vid'], vid_idx)
                filename_prefix_for_video = filename_prefix
                aux_str = '({}, time: {}-{})'.format(aux_info[vid_idx]['vid'],
                                                     str(datetime.timedelta(
                                                         seconds=aux_info[vid_idx]['start_time'])),
                                                     str(datetime.timedelta(
                                                         seconds=aux_info[vid_idx]['end_time'])))
                create_video_and_save(
                    save_path, 0, filename_prefix_for_video, 0,
                    target_seq[vid_idx].cpu().numpy(), out_dir_vec[vid_idx].cpu().numpy(),
                    np.reshape(self.data_loader['test_data_s2eg'].mean_dir_vec, -1), sentences[vid_idx],
                    audio=in_audio[vid_idx].cpu().numpy(), aux_str=aux_str,
                    clipping_to_shortest_stream=True, delete_audio_file=False)
        # loss
        beta = 0.1
        huber_loss = F.smooth_l1_loss(out_dir_vec / beta, target_poses / beta) * beta
        dis_output = self.s2eg_discriminator(out_dir_vec, in_emo_labels, in_text)
        gen_error = -torch.mean(torch.log(dis_output + 1e-8))
        kld = div_reg = None

        if (self.config_args.z_type == 'speaker' or self.config_args.z_type == 'random') and\
                self.config_args.loss_reg_weight > 0.0:
            if self.config_args.z_type == 'speaker':
                # enforcing divergent gestures btw original vid and other vid
                rand_idx = torch.randperm(vid_indices.shape[0])
                rand_vids = vid_indices[rand_idx]
            else:
                rand_vids = None

            out_dir_vec_rand_vid, z_rand_vid, _, _ = self.s2eg_generator(pre_seq, in_text, in_audio,
                                                                         in_emo_labels, rand_vids)
            beta = 0.05
            pose_l1 = F.smooth_l1_loss(out_dir_vec / beta, out_dir_vec_rand_vid.detach() / beta,
                                       reduction='none') * beta
            pose_l1 = pose_l1.sum(dim=1).sum(dim=1)

            pose_l1 = pose_l1.view(pose_l1.shape[0], -1).mean(1)
            z_l1 = F.l1_loss(z.detach(), z_rand_vid.detach(), reduction='none')
            z_l1 = z_l1.view(z_l1.shape[0], -1).mean(1)
            div_reg = -(pose_l1 / (z_l1 + 1.0e-5))
            div_reg = torch.clamp(div_reg, min=-1000)
            div_reg = div_reg.mean()

            if self.config_args.z_type == 'speaker':
                # speaker embedding KLD
                kld = -0.5 * torch.mean(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
                loss = self.config_args.loss_regression_weight * huber_loss +\
                    self.config_args.loss_kld_weight * kld +\
                    self.config_args.loss_reg_weight * div_reg
            else:
                loss = self.config_args.loss_regression_weight * huber_loss +\
                       self.config_args.loss_reg_weight * div_reg
        else:
            loss = self.config_args.loss_regression_weight * huber_loss  # + var_loss

        if self.meta_info['epoch'] > warm_up_epochs:
            loss += self.config_args.loss_gan_weight * gen_error

        if train:
            loss.backward()
            self.s2eg_gen_optimizer.step()

        loss_dict = {'loss': self.config_args.loss_regression_weight * huber_loss.item()}
        if kld:
            loss_dict['KLD'] = self.config_args.loss_kld_weight * kld.item()
        if div_reg:
            loss_dict['DIV_REG'] = self.config_args.loss_reg_weight * div_reg.item()

        if self.meta_info['epoch'] > warm_up_epochs and self.config_args.loss_gan_weight > 0.0:
            loss_dict['gen'] = self.config_args.loss_gan_weight * gen_error.item()
            loss_dict['dis'] = dis_error.item()
        loss_dict['total_loss'] = 0.
        for loss in loss_dict.keys():
            loss_dict['total_loss'] += loss_dict[loss]
        return loss_dict

    def per_train(self):

        batch_ser_loss = 0.
        batch_ser_accu = 0.
        batch_s2eg_loss = 0.
        num_batches = 0.

        for train_data_wav, train_labels_cat, train_labels_dim,\
                word_seq_tensor, word_seq_lengths, extended_word_seq,\
                pose_seq, vec_seq, audio, spectrogram, vid_indices in self.yield_batch(train=True):
            if self.args.train_ser:
                self.ser_model.train()
                ser_loss, train_labels_pred, train_labels_oh =\
                    self.forward_pass_ser(train_data_wav,
                                          train_labels_cat if self.args.emo_as_cats else train_labels_dim)
                ser_loss.backward()
                # nn.utils.clip_grad_norm_(self.ser_model.parameters(), self.args.gradient_clip)
                self.ser_optimizer.step()
                if torch.max(torch.abs(self.ser_model.linear3.weight.grad.data)) < 1e-10:
                    stop = 1
                if self.args.emo_as_cats:
                    train_accu = torch.sum((train_labels_cat - train_labels_pred) == 0) / len(train_labels_pred)
                else:
                    train_accu = - ser_loss.clone()

                # Compute statistics
                batch_ser_loss += ser_loss.item()
                batch_ser_accu += train_accu.item()

                self.iter_info['ser_loss'] = ser_loss.data.item()
                self.iter_info['ser_accu'] = train_accu.data.item()
                self.iter_info['lr_ser'] = '{:.6f}'.format(self.lr_ser)
                self.show_iter_info()
            else:
                self.ser_model.eval()
                with torch.no_grad():
                    _, train_labels_pred, train_labels_oh = self.forward_pass_ser(train_data_wav,
                                                                                  train_labels_cat)

            if self.args.train_s2eg:
                self.s2eg_generator.train()
                self.s2eg_discriminator.train()
                loss_dict = self.forward_pass_s2eg(extended_word_seq, audio, train_labels_oh,
                                                   vec_seq, vid_indices, train=True)
                # Compute statistics
                batch_s2eg_loss += loss_dict['total_loss']

                self.iter_info['s2eg_loss'] = loss_dict['total_loss']
                self.iter_info['lr_gen'] = '{}'.format(self.lr_s2eg_gen)
                self.iter_info['lr_dis'] = '{}'.format(self.lr_s2eg_dis)
                self.show_iter_info()

            self.meta_info['iter'] += 1
            num_batches += 1

        if self.args.train_ser:
            batch_ser_loss /= num_batches
            batch_ser_accu /= num_batches
            self.epoch_info['mean_ser_accu'] = batch_ser_accu
            self.epoch_info['mean_ser_loss'] = batch_ser_loss

        if self.args.train_s2eg:
            batch_s2eg_loss /= num_batches
            self.epoch_info['mean_s2eg_loss'] = batch_s2eg_loss

        self.show_epoch_info()
        self.io.print_timer()

        if self.args.train_ser:
            self.adjust_lr_ser()
        if self.args.train_s2eg:
            self.adjust_lr_s2eg()

    def per_eval(self):

        batch_ser_loss = 0.
        batch_ser_accu = 0.
        batch_s2eg_loss = 0.
        num_batches = 0.

        for eval_data_wav, eval_labels_cat, eval_labels_dim,\
            word_seq_tensor, word_seq_lengths, extended_word_seq, \
                pose_seq, vec_seq, audio, spectrogram, vid_indices in self.yield_batch(train=False):
            self.ser_model.eval()
            with torch.no_grad():
                ser_loss, eval_labels_pred, eval_labels_oh =\
                    self.forward_pass_ser(eval_data_wav,
                                          eval_labels_cat if self.args.emo_as_cats else eval_labels_dim)
                if self.args.emo_as_cats:
                    eval_accu = torch.sum((eval_labels_cat - eval_labels_pred) == 0) / len(eval_labels_pred)
                else:
                    eval_accu = - ser_loss.clone()

                if self.args.train_ser:
                    # Compute statistics
                    batch_ser_loss += ser_loss.item()
                    batch_ser_accu += eval_accu.item()

                    self.iter_info['ser_loss'] = ser_loss.data.item()
                    self.iter_info['ser_accu'] = eval_accu.data.item()
                    self.iter_info['lr_ser'] = '{:.6f}'.format(self.lr_ser)
                    self.show_iter_info()

            if self.args.train_s2eg:
                self.s2eg_generator.eval()
                self.s2eg_discriminator.eval()
                with torch.no_grad():
                    loss_dict = self.forward_pass_s2eg(extended_word_seq, audio, eval_labels_oh,
                                                       vec_seq, vid_indices, train=False)
                    # Compute statistics
                    batch_s2eg_loss += loss_dict['total_loss']

                    self.iter_info['s2eg_loss'] = loss_dict['total_loss']
                    self.iter_info['lr_gen'] = '{:.6f}'.format(self.lr_ser)
                    self.iter_info['lr_dis'] = '{:.6f}'.format(self.lr_ser)
                    self.show_iter_info()

            self.meta_info['iter'] += 1
            num_batches += 1

        if self.args.train_ser:
            batch_ser_loss /= num_batches
            batch_ser_accu /= num_batches
            self.epoch_info['mean_ser_accu'] = batch_ser_accu
            self.epoch_info['mean_ser_loss'] = batch_ser_loss
            if self.epoch_info['mean_ser_accu'] > self.best_ser_accu and \
                    self.meta_info['epoch'] > self.min_train_epochs:
                self.best_ser_accu = self.epoch_info['mean_ser_accu']
                self.best_ser_accu_loss = self.epoch_info['mean_ser_loss']
                self.best_ser_accu_epoch = self.meta_info['epoch']
                self.ser_accu_updated = True
            else:
                self.ser_accu_updated = False

        if self.args.train_s2eg:
            batch_s2eg_loss /= num_batches
            self.epoch_info['mean_s2eg_loss'] = batch_s2eg_loss
            if self.epoch_info['mean_s2eg_loss'] < self.best_s2eg_loss and \
                    self.meta_info['epoch'] > self.min_train_epochs:
                self.best_s2eg_loss = self.epoch_info['mean_s2eg_loss']
                self.best_s2eg_loss_epoch = self.meta_info['epoch']
                self.s2eg_loss_updated = True
            else:
                self.s2eg_loss_updated = False

        self.show_epoch_info()
        self.io.print_timer()

    def train(self):

        if self.args.ser_load_last_best:
            ser_model_found = self.load_model_at_epoch('ser', epoch=self.args.ser_start_epoch)
            if not ser_model_found and self.args.ser_start_epoch is not 'best':
                print('Warning! Trying to load best known model for ser: '.format(self.args.ser_start_epoch),
                      end='')
                ser_model_found = self.load_model_at_epoch('ser', epoch='best')
                self.args.ser_start_epoch = self.best_ser_accu_epoch if ser_model_found else 0
                print('loaded.')
                if not ser_model_found:
                    print('Warning! Starting at epoch 0')
                    self.args.ser_start_epoch = 0
        else:
            self.args.ser_start_epoch = 0
        if self.args.train_ser:
            for epoch in range(self.args.ser_start_epoch, self.args.ser_num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('SER training epoch: {}'.format(epoch))
                self.per_train()
                self.io.print_log('Done.')

                # evaluation
                if (epoch % self.args.eval_interval == 0) or (
                        epoch + 1 == self.args.num_epoch):
                    self.io.print_log('SER eval epoch: {}'.format(epoch))
                    self.per_eval()
                    self.io.print_log('Done.')

                # save model and weights
                if self.ser_accu_updated or (epoch % self.args.save_interval == 0 and epoch > self.min_train_epochs):
                    torch.save({'ser_model_dict': self.ser_model.state_dict()},
                               j(self.args.work_dir_ser, 'epoch_{}_accu_{:.4f}_loss_{:.4f}_model.pth.tar'.
                                 format(epoch, self.epoch_info['mean_ser_accu'], self.epoch_info['mean_ser_loss'])))

        if self.args.train_s2eg:
            if self.args.s2eg_load_last_best:
                s2eg_model_found = self.load_model_at_epoch('s2eg', epoch=self.args.s2eg_start_epoch)
                if not s2eg_model_found and self.args.s2eg_start_epoch is not 'best':
                    print('Warning! Trying to load best known model for s2eg: '.format(self.args.s2eg_start_epoch),
                          end='')
                    s2eg_model_found = self.load_model_at_epoch('s2eg', epoch='best')
                    self.args.s2eg_start_epoch = self.best_s2eg_loss_epoch if s2eg_model_found else 0
                    print('loaded.')
                    if not s2eg_model_found:
                        print('Warning! Starting at epoch 0')
                        self.args.s2eg_start_epoch = 0
            else:
                self.args.s2eg_start_epoch = 0
            for epoch in range(self.args.s2eg_start_epoch, self.args.s2eg_num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('S2EG training epoch: {}'.format(epoch))
                self.per_train()
                self.io.print_log('Done.')

                # evaluation
                if (epoch % self.args.eval_interval == 0) or (
                        epoch + 1 == self.args.num_epoch):
                    self.io.print_log('S2EG eval epoch: {}'.format(epoch))
                    self.per_eval()
                    self.io.print_log('Done.')

                # save model and weights
                if self.s2eg_loss_updated or (epoch % self.args.save_interval == 0 and epoch > self.min_train_epochs):
                    torch.save({'gen_model_dict': self.s2eg_generator.state_dict(),
                                'dis_model_dict': self.s2eg_discriminator.state_dict()},
                               j(self.args.work_dir_s2eg, 'epoch_{}_loss_{:.4f}_model.pth.tar'.
                                 format(epoch, self.epoch_info['mean_s2eg_loss'])))

    def generate_motion(self, load_saved_model=True, samples_to_generate=10,
                        randomized=True, ser_epoch='best', s2eg_epoch='best'):

        if load_saved_model:
            ser_model_found = self.load_model_at_epoch('ser', epoch=ser_epoch)
            assert ser_model_found, print('Speech emotion recognition model not found')
            s2eg_model_found = self.load_model_at_epoch('s2eg', epoch=s2eg_epoch)
            assert s2eg_model_found, print('Speech to emotive gestures model not found')
        self.ser_model.eval()
        self.s2eg_generator.eval()
        self.s2eg_discriminator.eval()

        start_time = time.time()
        test_data_wav, test_labels_cat, test_labels_dim, words,\
            aux_info, word_seq_tensor, word_seq_lengths, extended_word_seq, \
            pose_seq, vec_seq, target_seq, audio, spectrogram, vid_indices = \
            self.return_batch([samples_to_generate], randomized=randomized)
        with torch.no_grad():
            ser_loss, eval_labels_pred, eval_labels_oh = \
                self.forward_pass_ser(test_data_wav,
                                      test_labels_cat if self.args.emo_as_cats else test_labels_dim)
            loss_dict = self.forward_pass_s2eg(extended_word_seq, audio, eval_labels_oh,
                                               vec_seq, vid_indices, train=False,
                                               target_seq=target_seq, words=words, aux_info=aux_info,
                                               save_path=self.args.video_save_path,
                                               make_video=True)
        end_time = time.time()
        print('Total time taken: {:.2f} secs.'.format(end_time - start_time))
