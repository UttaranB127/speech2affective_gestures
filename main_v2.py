import argparse
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

import loader_v2 as loader
import processor_v2 as processor

from os.path import join as jn

from parse_args import parse_args

warnings.filterwarnings('ignore')


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
parser.add_argument('-b', '--base-path', required=True, type=str, help='Root directory of data files')
parser.add_argument('-c', '--config', required=True, is_config_file=True, help='Config file path')
parser.add_argument('--dataset-s2ag', type=str, default='ted_db', metavar='D-S2G',
                    help='dataset to train and validate speech to emotive gestures (default: ted_db)')
parser.add_argument('--dataset-test', type=str, default='ted_db', metavar='D-TST',
                    help='dataset to test emotive gestures (options: ted_db, genea_challenge_2020)')
parser.add_argument('-dap', '--dataset-s2ag-already-processed',
                    help='Optional. Set to True if dataset has already been processed.' +
                         'If not, or if you are not sure, set it to False.',
                    type=str2bool, default=True)
parser.add_argument('--frame-drop', type=int, default=2, metavar='FD',
                    help='frame down-sample rate (default: 2)')
parser.add_argument('--train-s2ag', type=str2bool, default=True, metavar='T-s2ag',
                    help='train the s2ag model (default: False)')
parser.add_argument('--use-multiple-gpus', type=str2bool, default=True, metavar='T',
                    help='use multiple GPUs if available (default: True)')
parser.add_argument('--s2ag-load-last-best', type=str2bool, default=True, metavar='s2ag-LB',
                    help='load the most recent best model for s2ag (default: True)')
parser.add_argument('--batch-size', type=int, default=512, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='number of threads? (default: 4)')
parser.add_argument('--s2ag-start-epoch', type=int, default=290, metavar='s2ag-SE',
                    help='starting epoch of training of s2ag (default: 0)')
parser.add_argument('--s2ag-num-epoch', type=int, default=500, metavar='s2ag-NE',
                    help='number of epochs to train s2ag (default: 1000)')
# parser.add_argument('--window-length', type=int, default=1, metavar='WL',
#                     help='max number of past time steps to take as input to transformer decoder (default: 60)')
parser.add_argument('--base-tr', type=float, default=1., metavar='TR',
                    help='base teacher rate (default: 1.0)')
parser.add_argument('--step', type=list, default=0.05 * np.arange(20), metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--lr-s2ag-decay', type=float, default=0.999, metavar='LRD-s2ag',
                    help='learning rate decay for s2ag (default: 0.999)')
parser.add_argument('--gradient-clip', type=float, default=0.1, metavar='GC',
                    help='gradient clip threshold (default: 0.1)')
parser.add_argument('--nesterov', type=str2bool, default=True,
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
parser.add_argument('--val-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is validated (default: 1)')
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
data_path = jn(args.base_path, '..', 'data')
models_s2ag_path = jn(args.base_path, 'models', 's2ag_v2_mfcc_run_3')
args.data_path = data_path
randomized = False

s2ag_config_args = parse_args(args.config)

args.work_dir_s2ag = jn(models_s2ag_path, args.dataset_s2ag)
os.makedirs(args.work_dir_s2ag, exist_ok=True)

args.video_save_path = jn(args.base_path, 'outputs', args.dataset_test, 'videos_trimodal_style')
os.makedirs(args.video_save_path, exist_ok=True)
args.quantitative_save_path = jn(args.base_path, 'outputs', 'quantitative')
os.makedirs(args.quantitative_save_path, exist_ok=True)

train_data_ted, val_data_ted, test_data_ted = loader.load_ted_db_data(data_path, s2ag_config_args, args.train_s2ag)

data_loader = dict(train_data_s2ag=train_data_ted, val_data_s2ag=val_data_ted, test_data_s2ag=test_data_ted)
pose_dim = 27
coords = 3
audio_sr = 16000

pr = processor.Processor(args.base_path, args, s2ag_config_args, data_loader, pose_dim, coords, audio_sr)

if args.train_s2ag:
    pr.train()

# pr.generate_gestures(samples_to_generate=data_loader['test_data_s2ag'].n_samples,
#                      randomized=randomized, s2ag_epoch=310, make_video=False)

data_params = {}
check_duration = False
if args.dataset_test.lower() == 'ted_db':
    data_params = {'env_file': jn(data_path, 'ted_db/lmdb_test_s2ag_v2_cache_mfcc_14'),
                   'clip_duration_range': [5, 30],
                   'audio_sr': 16000}
    check_duration = True
elif args.dataset_test.lower() == 'genea_challenge_2020':
    data_params = {'data_path': jn(data_path, 'genea_challenge_2020/test')}

samples = ['5QTjSH1KGlY_760.32_767.79', 'cK74vhqzeeQ_683.22_695.35', 'E7oq6J8HvKw_216.23_232.37', 'hfznpykprP0_220.88_235.68',
           'luoKOkTxOtU_246.92_260.39', 'mLufqwmPl1A_646.12_653.59', 'sEOSCziWuP8_807.65_815.79', 'yF4MgSh7VO4_29.24_40.04']
pr.generate_gestures_by_dataset(dataset=args.dataset_test, data_params=data_params,
                                randomized=randomized, check_duration=check_duration,
                                s2ag_epoch=290, samples=samples, make_video=True, save_pkl=True)
