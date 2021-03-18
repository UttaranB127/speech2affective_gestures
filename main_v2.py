import argparse
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

import loader_v2 as loader
import processor_v2 as processor

from os.path import join as j

from config.parse_args import parse_args

warnings.filterwarnings('ignore')


base_path = os.path.dirname(os.path.realpath(__file__))
data_path = j(base_path, '../../data')

models_s2eg_path = j(base_path, 'models', 's2eg_v2')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Speech to Emotive Gestures')
parser.add_argument('--dataset-s2eg', type=str, default='ted_db', metavar='D-S2G',
                    help='dataset to train and evaluate speech to emotive gestures (default: ted)')
parser.add_argument('-dap', '--dataset-s2eg-already-processed',
                    help='Optional. Set to True if dataset has already been processed.' +
                         'If not, or if you are not sure, set it to False.',
                    type=str2bool, default=True)
parser.add_argument('-c', '--config', required=True, is_config_file=True, help='Config file path')
parser.add_argument('--frame-drop', type=int, default=2, metavar='FD',
                    help='frame down-sample rate (default: 2)')
parser.add_argument('--add-mirrored', type=bool, default=False, metavar='AM',
                    help='perform data augmentation by mirroring all the sequences (default: False)')
parser.add_argument('--train-s2eg', type=bool, default=True, metavar='T-S2EG',
                    help='train the s2eg model (default: True)')
parser.add_argument('--use-multiple-gpus', type=bool, default=True, metavar='T',
                    help='use multiple GPUs if available (default: True)')
parser.add_argument('--s2eg-load-last-best', type=bool, default=True, metavar='S2EG-LB',
                    help='load the most recent best model for s2eg (default: True)')
parser.add_argument('--batch-size', type=int, default=512, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='number of threads? (default: 4)')
parser.add_argument('--s2eg-start-epoch', type=int, default=142, metavar='S2EG-SE',
                    help='starting epoch of training of s2eg (default: 0)')
parser.add_argument('--s2eg-num-epoch', type=int, default=500, metavar='S2EG-NE',
                    help='number of epochs to train s2eg (default: 1000)')
# parser.add_argument('--window-length', type=int, default=1, metavar='WL',
#                     help='max number of past time steps to take as input to transformer decoder (default: 60)')
parser.add_argument('--base-tr', type=float, default=1., metavar='TR',
                    help='base teacher rate (default: 1.0)')
parser.add_argument('--step', type=list, default=0.05 * np.arange(20), metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--lr-s2eg-decay', type=float, default=0.999, metavar='LRD-S2EG',
                    help='learning rate decay for s2eg (default: 0.999)')
parser.add_argument('--gradient-clip', type=float, default=0.1, metavar='GC',
                    help='gradient clip threshold (default: 0.1)')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--upper-body-weight', type=float, default=1., metavar='UBW',
                    help='loss weight on the upper body joint motions (default: 2.05)')
parser.add_argument('--affs-reg', type=float, default=0.8, metavar='AR',
                    help='regularization for affective features loss (default: 0.01)')
parser.add_argument('--quat-norm-reg', type=float, default=0.1, metavar='QNR',
                    help='regularization for unit norm constraint (default: 0.01)')
parser.add_argument('--quat-reg', type=float, default=1.2, metavar='QR',
                    help='regularization for quaternion loss (default: 0.01)')
parser.add_argument('--recons-reg', type=float, default=1.2, metavar='RCR',
                    help='regularization for reconstruction loss (default: 1.2)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--save-interval', type=int, default=10, metavar='SI',
                    help='interval after which model is saved (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pavi-log', action='store_true', default=False,
                    help='pavi log')
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')
# TO ADD: save_result

args = parser.parse_args()
args.data_path = data_path
randomized = False

s2eg_config_args = parse_args()

args.work_dir_s2eg = j(models_s2eg_path, args.dataset_s2eg)
os.makedirs(args.work_dir_s2eg, exist_ok=True)

args.video_save_path = j(base_path, 'outputs', 'videos_trimodal_style')
os.makedirs(args.video_save_path, exist_ok=True)
args.quantitative_save_path = j(base_path, 'outputs', 'quantitative')
os.makedirs(args.quantitative_save_path, exist_ok=True)

train_data_ted, eval_data_ted, test_data_ted = loader.load_ted_db_data(data_path, s2eg_config_args)

data_loader = dict(train_data_s2eg=train_data_ted, eval_data_s2eg=eval_data_ted, test_data_s2eg=test_data_ted)
pose_dim = 27
coords = 3
audio_sr = 16000

pr = processor.Processor(args, s2eg_config_args, data_loader, pose_dim, coords, audio_sr)

if args.train_s2eg:
    pr.train()

# pr.generate_gestures(samples_to_generate=len(data_loader['test_data_s2eg_wav']),
#                      randomized=randomized, ser_epoch='best', s2eg_epoch=142)

# pr.generate_gestures_by_env_file(j(data_path, 'ted_db/lmdb_test'), [5, 12],
#                                  randomized=randomized, ser_epoch='best', s2eg_epoch=142)

# generate a random audio file
# from scipy.io.wavfile import write
# data = np.random.uniform(-1, 1, 44100)
# scaled = np.int16(data/np.max(np.abs(data)) * 32767)
# write('test.wav', 44100, scaled)

# for k in range(1):
#     with data_s2eg.lmdb_env.begin(write=False) as txn:
#         key = '{:010}'.format(k).encode('ascii')
#         sample = txn.get(key)
#         sample = pyarrow.deserialize(sample)
#         word_seq, pose_seq, vec_seq, audio, spectrogram, mfcc_features, aux_info = sample
# half = self.num_train_samples // 2
# word_seq_part_1 = dict.fromkeys([str(k).zfill(6) for k in range(half)])
# pose_seq_part_1 = np.zeros((half, pose_seq.shape[0], pose_seq.shape[1], pose_seq.shape[2]))
# vec_seq_part_1 = np.zeros((half, vec_seq.shape[0], vec_seq.shape[1], vec_seq.shape[2]))
# audio_part_1 = np.zeros((half, audio.shape[0]), dtype=np.int16)
# audio_max_part_1 = np.zeros(half)
# mfcc_features_part_1 = np.zeros((half, mfcc_features.shape[0], mfcc_features.shape[1]))
# aux_info_part_1 = dict.fromkeys([str(k).zfill(6) for k in range(half)])
# for k in range(half):
#     with data_s2eg.lmdb_env.begin(write=False) as txn:
#         key = '{:010}'.format(k).encode('ascii')
#         sample = txn.get(key)
#         sample = pyarrow.deserialize(sample)
#         word_seq, pose_seq, vec_seq, audio, spectrogram, mfcc_features, aux_info = sample
#     pose_seq_part_1[k] = pose_seq
#     vec_seq_part_1[k] = vec_seq
#     audio_max_part_1[k] = np.max(np.abs(audio))
#     audio_part_1[k] = np.int16(audio/audio_max_part_1[k] * 32767)
#     mfcc_features_part_1[k] = mfcc_features
#
#     word_seq_part_1[str(k).zfill(6)] = word_seq
#     aux_info_part_1[str(k).zfill(6)] = aux_info
#     print('\rstored key {}'.format(k), end='')
# print()
# save_dir = jn('../../data/ted_db/individual/train')
# os.makedirs(save_dir, exist_ok=True)
# np.savez_compressed(jn(save_dir, 'part_1.npz'), word_seq=word_seq_part_1, pose_seq=pose_seq_part_1,
#                     vec_seq=vec_seq_part_1, audio=audio_part_1, audio_max=audio_max_part_1,
#                     mfcc_features=mfcc_features_part_1, aux_info=aux_info_part_1)
# print('done')


# for k in range(1):
#     with data_s2eg.lmdb_env.begin(write=False) as txn:
#         key = '{:010}'.format(k).encode('ascii')
#         sample = txn.get(key)
#         sample = pyarrow.deserialize(sample)
#         word_seq, pose_seq, vec_seq, audio, spectrogram, mfcc_features, aux_info = sample
# half = self.num_train_samples // 2
# pose_seq_part_2 = np.zeros((half, pose_seq.shape[0], pose_seq.shape[1], pose_seq.shape[2]))
# vec_seq_part_2 = np.zeros((half, vec_seq.shape[0], vec_seq.shape[1], vec_seq.shape[2]))
# audio_part_2 = np.zeros((half, audio.shape[0]), dtype=np.int16)
# audio_max_part_2 = np.zeros(half)
# spectrogram_part_2 = np.zeros((half, spectrogram.shape[0], spectrogram.shape[1]))
# mfcc_features_part_2 = np.zeros((half, mfcc_features.shape[0], mfcc_features.shape[1]))
# word_seq_part_2 = dict.fromkeys([str(k).zfill(6) for k in range(half)])
# aux_info_part_2 = dict.fromkeys([str(k).zfill(6) for k in range(half)])
# for k in range(half):
#     with data_s2eg.lmdb_env.begin(write=False) as txn:
#         key = '{:010}'.format(k).encode('ascii')
#         sample = txn.get(key)
#         sample = pyarrow.deserialize(sample)
#         word_seq, pose_seq, vec_seq, audio, spectrogram, mfcc_features, aux_info = sample
#     pose_seq_part_2[k - half] = pose_seq
#     vec_seq_part_2[k - half] = vec_seq
#     audio_max_part_2[k - half] = np.max(np.abs(audio))
#     audio_part_2[k - half] = np.int16(audio/audio_max_part_2[k - half] * 32767)
#     spectrogram_part_2[k - half] = spectrogram
#     mfcc_features_part_2[k - half] = mfcc_features
#
#     word_seq_part_2[str(k - half).zfill(6)] = word_seq
#     aux_info_part_2[str(k - half).zfill(6)] = aux_info
#     print('\rstored key {}'.format(k), end='')
# print()
# save_dir = jn('../../data/ted_db/individual/train')
# os.makedirs(save_dir, exist_ok=True)
# np.savez_compressed(jn(save_dir, 'part_2.npz'), word_seq=word_seq_part_2, pose_seq=pose_seq_part_2,
#                     vec_seq=vec_seq_part_2, audio=audio_part_2, audio_max=audio_max_part_2,
#                     mfcc_features=mfcc_features_part_2, aux_info=aux_info_part_2)
# print('done')


# data_s2eg = self.data_loader['train_data_s2eg']
# for k in range(1):
#     with data_s2eg.lmdb_env.begin(write=False) as txn:
#         key = '{:010}'.format(k).encode('ascii')
#         sample = txn.get(key)
#         sample = pyarrow.deserialize(sample)
#         word_seq, pose_seq, vec_seq, audio, spectrogram, mfcc_features, aux_info = sample
#
# word_seq_all = dict.fromkeys([str(k).zfill(6) for k in range(self.num_train_samples)])
# pose_seq_all = np.zeros((self.num_train_samples, pose_seq.shape[0], pose_seq.shape[1], pose_seq.shape[2]))
# vec_seq_all = np.zeros((self.num_train_samples, vec_seq.shape[0], vec_seq.shape[1], vec_seq.shape[2]))
# audio_all = np.zeros((self.num_train_samples, audio.shape[0]), dtype=np.int16)
# audio_max_all = np.zeros(self.num_train_samples)
# mfcc_features_all = np.zeros((self.num_train_samples, mfcc_features.shape[0], mfcc_features.shape[1]))
# aux_info_all = dict.fromkeys([str(k).zfill(6) for k in range(self.num_train_samples)])
# for k in range(self.num_train_samples):
#     with data_s2eg.lmdb_env.begin(write=False) as txn:
#         key = '{:010}'.format(k).encode('ascii')
#         sample = txn.get(key)
#         sample = pyarrow.deserialize(sample)
#         word_seq, pose_seq, vec_seq, audio, spectrogram, mfcc_features, aux_info = sample
#     pose_seq_all[k] = pose_seq
#     vec_seq_all[k] = vec_seq
#     audio_max_all[k] = np.max(np.abs(audio))
#     audio_all[k] = np.int16(audio/audio_max_all[k] * 32767)
#     mfcc_features_all[k] = mfcc_features
#
#     word_seq_all[str(k).zfill(6)] = word_seq
#     aux_info_all[str(k).zfill(6)] = aux_info
#     print('\rstored key {}'.format(k), end='')
# print()
# save_dir = jn('../../data/ted_db/npz')
# os.makedirs(save_dir, exist_ok=True)
# np.savez_compressed(jn(save_dir, 'train.npz'), word_seq=word_seq_all, pose_seq=pose_seq_all,
#                     vec_seq=vec_seq_all, audio=audio_all, audio_max=audio_max_all,
#                     mfcc_features=mfcc_features_all, aux_info=aux_info_all)
# print('done')


# data_s2eg = self.data_loader['eval_data_s2eg']
# for k in range(1):
#     with data_s2eg.lmdb_env.begin(write=False) as txn:
#         key = '{:010}'.format(k).encode('ascii')
#         sample = txn.get(key)
#         sample = pyarrow.deserialize(sample)
#         word_seq, pose_seq, vec_seq, audio, spectrogram, mfcc_features, aux_info = sample
#
# word_seq_all = dict.fromkeys([str(k).zfill(6) for k in range(self.num_eval_samples)])
# pose_seq_all = np.zeros((self.num_eval_samples, pose_seq.shape[0], pose_seq.shape[1], pose_seq.shape[2]))
# vec_seq_all = np.zeros((self.num_eval_samples, vec_seq.shape[0], vec_seq.shape[1], vec_seq.shape[2]))
# audio_all = np.zeros((self.num_eval_samples, audio.shape[0]), dtype=np.int16)
# audio_max_all = np.zeros(self.num_eval_samples)
# mfcc_features_all = np.zeros((self.num_eval_samples, mfcc_features.shape[0], mfcc_features.shape[1]))
# aux_info_all = dict.fromkeys([str(k).zfill(6) for k in range(self.num_eval_samples)])
# for k in range(self.num_eval_samples):
#     with data_s2eg.lmdb_env.begin(write=False) as txn:
#         key = '{:010}'.format(k).encode('ascii')
#         sample = txn.get(key)
#         sample = pyarrow.deserialize(sample)
#         word_seq, pose_seq, vec_seq, audio, spectrogram, mfcc_features, aux_info = sample
#     pose_seq_all[k] = pose_seq
#     vec_seq_all[k] = vec_seq
#     audio_max_all[k] = np.max(np.abs(audio))
#     audio_all[k] = np.int16(audio/audio_max_all[k] * 32767)
#     mfcc_features_all[k] = mfcc_features
#
#     word_seq_all[str(k).zfill(6)] = word_seq
#     aux_info_all[str(k).zfill(6)] = aux_info
#     print('\rstored key {}'.format(k), end='')
# print()
# save_dir = jn('../../data/ted_db/npz')
# os.makedirs(save_dir, exist_ok=True)
# np.savez_compressed(jn(save_dir, 'eval.npz'), word_seq=word_seq_all, pose_seq=pose_seq_all,
#                     vec_seq=vec_seq_all, audio=audio_all, audio_max=audio_max_all,
#                     mfcc_features=mfcc_features_all, aux_info=aux_info_all)
# print('done')
