# sys
import glob
import librosa
import lmdb
import multiprocessing
import numpy as np
import os
import pickle
import pyarrow
import python_speech_features as ps
import pyttsx3
import re
import wave

import utils.constant as constant

from joblib import Parallel, delayed
from nltk.stem.porter import PorterStemmer
from os.path import join as j
from scipy.io import wavfile
from tqdm import tqdm

from utils.data_preprocessor import DataPreprocessor
from utils.ted_db_utils import calc_spectrogram_length_from_motion_length
from utils.vocab import Vocab
from utils.vocab_utils import build_vocab


emotions_names_10_cats = ['neu', 'hap', 'exc', 'sur', 'fea', 'sad', 'dis', 'ang', 'fru', 'oth']
emotions_names_07_cats = ['neu', 'hap', 'fea', 'sad', 'dis', 'ang', 'oth']
nrc_vad_lexicon_file = '../../data/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt'
nrc_vad_lexicon = {}

with open(nrc_vad_lexicon_file, 'r') as nf:
    heading = nf.readline()
    lines = nf.readlines()
    for line in lines:
        line_split = line.split('\t')
        lexeme = line_split[0]
        v = float(line_split[1])
        a = float(line_split[2])
        d = float(line_split[3].split('\n')[0])
        nrc_vad_lexicon[lexeme] = np.array([v, a, d])
porter_stemmer = PorterStemmer()

tts_engine = pyttsx3.init()


def get_vad(lexeme_raw):
    lexeme_lower = lexeme_raw.lower()
    lexeme_stemmed = porter_stemmer.stem(lexeme_lower)
    if lexeme_lower in nrc_vad_lexicon.keys():
        return nrc_vad_lexicon[lexeme_lower]
    if lexeme_stemmed in nrc_vad_lexicon.keys():
        return nrc_vad_lexicon[lexeme_stemmed]
    return np.zeros(3)


def record_and_load_audio(audio_file, text, rate, trimmed=False):
    tts_engine.setProperty('rate', rate)
    tts_engine.save_to_file(text, audio_file)
    tts_engine.runAndWait()
    fs, audio_data = wavfile.read(audio_file)
    audio_data = np.trim_zeros(audio_data)
    if trimmed:
        audio_data = np.trim_zeros(audio_data)
    return fs, audio_data


def get_gesture_splits(sentence, words, num_frames, fps):
    audio_file = 'temp.mp3'
    best_rate = 50
    least_diff = np.inf
    for rate in range(50, 200):
        fs, audio_data = record_and_load_audio(audio_file, sentence, rate, trimmed=True)
        diff = np.abs(len(audio_data) / fs - num_frames / fps)
        if diff < least_diff:
            least_diff = np.copy(diff)
            best_rate = np.copy(rate)
        elif diff > least_diff:
            break
    fs, audio_data = record_and_load_audio(audio_file, sentence, best_rate, trimmed=True)
    sentence_frames = len(audio_data)
    word_frames = []
    fs_s = []
    total_word_frames = 0
    for word in words:
        if len(word) > 0:
            fs, audio_data = record_and_load_audio(audio_file, word, best_rate, trimmed=True)
            fs_s.append(fs)
            word_frames.append(len(audio_data))
            total_word_frames += len(audio_data)
    sampling_ratio = sentence_frames / total_word_frames
    splits = [0]
    for fs, w in zip(fs_s, word_frames):
        splits.append(int(np.ceil(splits[-1] + w * sampling_ratio * fps / fs)))

    if os.path.exists(audio_file):
        os.remove(audio_file)

    return int(best_rate), splits


def split_data_dict(data_dict, eval_size=0.1, randomized=True, fill=1):
    num_samples = len(data_dict)
    num_samples_eval = int(round(eval_size * num_samples))
    samples_all = np.array(list(data_dict.keys()), dtype=int)
    if randomized:
        samples_eval = np.random.choice(samples_all, num_samples_eval, replace=False)
    else:
        # samples_eval = samples_all[-num_samples_eval:]
        samples_eval = np.loadtxt('samples_eval.txt').astype(int)
    samples_train = np.setdiff1d(samples_all, samples_eval)
    data_dict_train = dict()
    data_dict_eval = dict()
    for idx, sample_idx in enumerate(samples_train):
        data_dict_train[str(idx).zfill(fill)] = data_dict[str(sample_idx).zfill(fill)]
    for idx, sample_idx in enumerate(samples_eval):
        data_dict_eval[str(idx).zfill(fill)] = data_dict[str(sample_idx).zfill(fill)]
    return data_dict_train, data_dict_eval


def to_one_hot(categorical_value, categories):
    index = categories.index(categorical_value)
    one_hot_array = np.zeros(len(categories))
    one_hot_array[index] = 1.
    return one_hot_array


def read_wav_file(file_name):
    file = wave.open(file_name, 'r')
    params = file.getparams()
    num_channels, sample_width, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wave_data = np.fromstring(str_data, dtype=np.short)
    # wave_data = np.float(wave_data*1.0/max(abs(wave_data)))  # normalization)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wave_data, time, framerate


def load_data():
    f = open('z_score_40.pkl', 'rb')
    mean1, std1, mean2, std2, mean3, std3 = pickle.load(f)
    return mean1, std1, mean2, std2, mean3, std3


def extract_07_categorical_emotions(string):
    if string == 'exc' or string == 'sur':
        string = 'hap'
    if string == 'fru':
        string = 'ang'
    if string == 'xxx':
        string = 'oth'
    emotions_cat = np.zeros(len(emotions_names_07_cats), dtype=int)
    emotions_cat[emotions_names_07_cats.index(string)] = 1
    return emotions_cat


def extract_10_categorical_emotions(string):
    if string == 'xxx':
        string = 'oth'
    emotions_cat = np.zeros(len(emotions_names_10_cats), dtype=int)
    emotions_cat[emotions_names_10_cats.index(string)] = 1
    return emotions_cat


def extract_dimensional_emotions(string):
    # a: dimensional emotion, c: categorical emotion
    # e: evaluator, f/m: self-reported
    if string[:3].lower() == 'a-e':
        emotions_dim = string.split()
        emotions_dim = [0. if emotions_dim[i] == ';'
                    else float(emotions_dim[i].split(';')[0])
                    for i in [2, 4, 6]]
        return emotions_dim
    return []


def append_idx(idx_list, data_count, time, block_size):
    if time <= block_size:
        idx_list.append(data_count - 1)
    else:
        # idx_list.append(data_count - 2)
        idx_list.append(data_count - 1)


def load_iemocap_data(data_dir, dataset, dimensional_min=-0., dimensional_max=6.,
                      block_size=300, filter_num=40, epsilon=1e-5):
    dataset_dir = j(data_dir, dataset)
    processed_dir = j(dataset_dir, 'processed_07_cats')
    os.makedirs(processed_dir, exist_ok=True)
    train_data_wav_file = j(processed_dir, 'train_data_wav.npz')
    eval_data_wav_file = j(processed_dir, 'eval_data_wav.npz')
    test_data_wav_file = j(processed_dir, 'test_data_wav.npz')
    train_labels_cat_file = j(processed_dir, 'train_labels_cat.npz')
    eval_labels_cat_file = j(processed_dir, 'eval_labels_cat.npz')
    test_labels_cat_file = j(processed_dir, 'test_labels_cat.npz')
    train_labels_dim_file = j(processed_dir, 'train_labels_dim.npz')
    eval_labels_dim_file = j(processed_dir, 'eval_labels_dim.npz')
    test_labels_dim_file = j(processed_dir, 'test_labels_dim.npz')
    stats_file = j(processed_dir, 'stats.pkl')

    if not (os.path.exists(train_data_wav_file)
            and os.path.exists(eval_data_wav_file)
            and os.path.exists(test_data_wav_file)
            and os.path.exists(train_labels_cat_file)
            and os.path.exists(eval_labels_cat_file)
            and os.path.exists(test_labels_cat_file)
            and os.path.exists(train_labels_dim_file)
            and os.path.exists(eval_labels_dim_file)
            and os.path.exists(test_labels_dim_file)
            and os.path.exists(stats_file)):

        session_set_train = [1, 2, 3, 4]
        session_set_test = [5]
        data_wav_list_1 = []
        data_wav_list_2 = []
        data_wav_list_3 = []
        labels_cat_list = []
        labels_dim_list = []
        data_count = 0
        train_idx = []
        eval_idx = []
        test_idx = []
        print('--------: -------------- (-- of --). Part -- of --. Total data size: ------', end='')

        # sessions 1, 2, 3, 4, 5
        session_dirs = glob.glob(j(dataset_dir, 'Session*'))
        for session in session_dirs:
            session_name = session.split('/')[-1]
            wav_dir = j(dataset_dir, session, 'sentences/wav')
            emo_dir = j(dataset_dir, session, 'dialog/EmoEvaluation')
            num_sessions = len(os.listdir(wav_dir))
            for sess_idx, sess in enumerate(os.listdir(wav_dir)):
                if 'impro' not in sess:
                    continue
                # impro: improvisation, script: scripted
                emo_file = j(emo_dir, sess + '.txt')
                emotions_cat = []
                emotions_dim = []
                with open(emo_file, 'r') as ef:
                    ef_lines = ef.readlines()
                    for ef_line in ef_lines:
                        if ef_line[0] == '[':
                            emotions_cat.append(extract_07_categorical_emotions(ef_line.split()[4]))
                            emotions_dim.append([float(x) for x in re.findall('\d+\.\d+', ef_line)[-3:]])

                        # extract_dimensional_emotions(ef_line)

                wav_files = glob.glob(j(wav_dir, sess, '*.wav'))
                num_wav_files = len(wav_files)
                assert num_wav_files == len(emotions_cat), 'Number of annotations do not match number of .wav files'
                assert num_wav_files == len(emotions_dim), 'Number of annotations do not match number of .wav files'
                for wav_idx, wav_file_name in enumerate(wav_files):
                    data, time, rate = read_wav_file(wav_file_name)
                    mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
                    delta1 = ps.delta(mel_spec, 2)
                    delta2 = ps.delta(delta1, 2)

                    time = mel_spec.shape[0]
                    if time <= block_size:
                        part = mel_spec
                        delta11 = delta1
                        delta21 = delta2
                        part = np.pad(part, ((0, block_size - part.shape[0]), (0, 0)), 'constant',
                                      constant_values=0)
                        delta11 = np.pad(delta11, ((0, block_size - delta11.shape[0]), (0, 0)), 'constant',
                                         constant_values=0)
                        delta21 = np.pad(delta21, ((0, block_size - delta21.shape[0]), (0, 0)), 'constant',
                                         constant_values=0)
                        # train_data_1[train_num * block_size:(train_num + 1) * block_size] = part
                        # train_data_2[train_num * block_size:(train_num + 1) * block_size] = delta11
                        # train_data_3[train_num * block_size:(train_num + 1) * block_size] = delta21
                        data_wav_list_1.append(part.tolist())
                        data_wav_list_2.append(delta11.tolist())
                        data_wav_list_3.append(delta21.tolist())

                        labels_cat_list.append(emotions_cat[wav_idx])
                        labels_dim_list.append(emotions_dim[wav_idx])

                        data_count += 1
                    else:
                        for begin in np.arange(0, time, 100):
                            end = begin + block_size
                            end_from_last = time - begin
                            begin_from_last = end_from_last - block_size
                            if end > time:
                                break

                            part = mel_spec[begin:end, :]
                            delta11 = delta1[begin:end, :]
                            delta21 = delta2[begin:end, :]
                            part_from_last = mel_spec[begin_from_last:end_from_last, :]
                            delta11_from_last = delta1[begin_from_last:end_from_last, :]
                            delta21_from_last = delta2[begin_from_last:end_from_last, :]

                            data_wav_list_1.append(part.tolist())
                            data_wav_list_2.append(delta11.tolist())
                            data_wav_list_3.append(delta21.tolist())

                            labels_cat_list.append(emotions_cat[wav_idx])
                            labels_dim_list.append(emotions_dim[wav_idx])

                            data_count += 1
                            # data_wav_list_1.append(part_from_last.tolist())
                            # data_wav_list_2.append(delta11_from_last.tolist())
                            # data_wav_list_3.append(delta21_from_last.tolist())
                            # data_count += 2
                    print('\r{}: {} ({:d} of {:d}). Part {:d} of {:d}. Total data size: {:d}'
                          .format(session_name, sess, sess_idx + 1, num_sessions,
                                  wav_idx + 1, num_wav_files, data_count), end='')
                    if int(session[-1]) in session_set_train:
                        append_idx(train_idx, data_count, time, block_size)
                    elif int(session[-1]) in session_set_test:
                        if wav_file_name.split('/')[-1][-8] == 'M':
                            append_idx(test_idx, data_count, time, block_size)
                        else:
                            append_idx(eval_idx, data_count, time, block_size)

        print()
        train_data_wav_1 = np.array([data_wav_list_1[i] for i in train_idx])
        train_data_wav_2 = np.array([data_wav_list_2[i] for i in train_idx])
        train_data_wav_3 = np.array([data_wav_list_3[i] for i in train_idx])

        eval_data_wav_1 = np.array([data_wav_list_1[i] for i in eval_idx])
        eval_data_wav_2 = np.array([data_wav_list_2[i] for i in eval_idx])
        eval_data_wav_3 = np.array([data_wav_list_3[i] for i in eval_idx])

        test_data_wav_1 = np.array([data_wav_list_1[i] for i in test_idx])
        test_data_wav_2 = np.array([data_wav_list_2[i] for i in test_idx])
        test_data_wav_3 = np.array([data_wav_list_3[i] for i in test_idx])

        train_labels_cat = np.array([labels_cat_list[i] for i in train_idx])
        eval_labels_cat = np.array([labels_cat_list[i] for i in eval_idx])
        test_labels_cat = np.array([labels_cat_list[i] for i in test_idx])

        train_labels_dim = \
            (np.array([labels_dim_list[i] for i in train_idx]) - dimensional_min) / (dimensional_max - dimensional_min)
        eval_labels_dim = \
            (np.array([labels_dim_list[i] for i in eval_idx]) - dimensional_min) / (dimensional_max - dimensional_min)
        test_labels_dim = \
            (np.array([labels_dim_list[i] for i in test_idx]) - dimensional_min) / (dimensional_max - dimensional_min)

        # mean1 = np.mean(train_data_wav_1, axis=(0, 1))
        # std1 = np.std(train_data_wav_1, axis=(0, 1))
        # mean2 = np.mean(train_data_wav_2, axis=(0, 1))
        # std2 = np.std(train_data_wav_2, axis=(0, 1))
        # mean3 = np.mean(train_data_wav_3, axis=(0, 1))
        # std3 = np.std(train_data_wav_3, axis=(0, 1))
        # train_data_wav = np.moveaxis(np.array([(train_data_wav_1 - mean1) / (std1 + epsilon),
        #                                        (train_data_wav_2 - mean2) / (std2 + epsilon),
        #                                        (train_data_wav_3 - mean3) / (std3 + epsilon)]),
        #                              0, 1)
        # eval_data_wav = np.moveaxis(np.array([(eval_data_wav_1 - mean1) / (std1 + epsilon),
        #                                       (eval_data_wav_2 - mean2) / (std2 + epsilon),
        #                                       (eval_data_wav_3 - mean3) / (std3 + epsilon)]),
        #                             0, 1)
        # test_data_wav = np.moveaxis(np.array([(test_data_wav_1 - mean1) / (std1 + epsilon),
        #                                       (test_data_wav_2 - mean2) / (std2 + epsilon),
        #                                       (test_data_wav_3 - mean3) / (std3 + epsilon)]),
        #                             0, 1)

        max1 = np.max(train_data_wav_1)
        min1 = np.min(train_data_wav_1)
        max2 = np.max(train_data_wav_2)
        min2 = np.min(train_data_wav_2)
        max3 = np.max(train_data_wav_3)
        min3 = np.min(train_data_wav_3)
        train_data_wav = np.moveaxis(np.array([(train_data_wav_1 - min1) / (max1 - min1),
                                               (train_data_wav_2 - min2) / (max2 - min2),
                                               (train_data_wav_3 - min3) / (max3 - min3)]),
                                     0, 1)
        eval_data_wav = np.moveaxis(np.array([(eval_data_wav_1 - min1) / (max1 - min1),
                                              (eval_data_wav_2 - min2) / (max2 - min2),
                                              (eval_data_wav_3 - min3) / (max3 - min3)]),
                                    0, 1)
        test_data_wav = np.moveaxis(np.array([(test_data_wav_1 - min1) / (max1 - min1),
                                              (test_data_wav_2 - min2) / (max2 - min2),
                                              (test_data_wav_3 - min3) / (max3 - min3)]),
                                    0, 1)

        np.savez_compressed(train_data_wav_file, train_data_wav)
        print('Successfully saved wave train data.')
        np.savez_compressed(eval_data_wav_file, eval_data_wav)
        print('Successfully saved wave eval data.')
        np.savez_compressed(test_data_wav_file, test_data_wav)
        print('Successfully saved wave test data.')

        np.savez_compressed(train_labels_cat_file, train_labels_cat)
        print('Successfully saved categorical train labels.')
        np.savez_compressed(eval_labels_cat_file, eval_labels_cat)
        print('Successfully saved categorical eval labels.')
        np.savez_compressed(test_labels_cat_file, test_labels_cat)
        print('Successfully saved categorical test labels.')

        np.savez_compressed(train_labels_dim_file, train_labels_dim)
        print('Successfully saved dimensional train labels.')
        np.savez_compressed(eval_labels_dim_file, eval_labels_dim)
        print('Successfully saved dimensional eval labels.')
        np.savez_compressed(test_labels_dim_file, test_labels_dim)
        print('Successfully saved dimensional test labels.')

        # with open(stats_file, 'wb') as af:
        #     pickle.dump((mean1, std1, mean2, std2, mean3, std3), af)
        # means = np.array([mean1, mean2, mean3])
        # stds = np.array([std1, std2, std3])
        with open(stats_file, 'wb') as af:
            pickle.dump((max1, min1, max2, min2, max3, min3), af)
        means = np.array([max1, max2, max3])
        stds = np.array([min1, min2, min3])
        print('Successfully saved stats.')
    else:
        train_data_wav = np.load(train_data_wav_file)['arr_0']
        eval_data_wav = np.load(eval_data_wav_file)['arr_0']
        test_data_wav = np.load(test_data_wav_file)['arr_0']

        train_labels_cat = np.load(train_labels_cat_file)['arr_0']
        eval_labels_cat = np.load(eval_labels_cat_file)['arr_0']
        test_labels_cat = np.load(test_labels_cat_file)['arr_0']

        train_labels_dim = np.load(train_labels_dim_file)['arr_0']
        eval_labels_dim = np.load(eval_labels_dim_file)['arr_0']
        test_labels_dim = np.load(test_labels_dim_file)['arr_0']

        with open(stats_file, 'rb') as af:
            stats = pickle.load(af)
        means = np.array(stats[:3])
        stds = np.array(stats[4:])

    return train_data_wav, eval_data_wav, test_data_wav, \
        train_labels_cat, eval_labels_cat, test_labels_cat, \
        train_labels_dim, eval_labels_dim, test_labels_dim, \
        means, stds


class TedDBParams:
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec,
                 speaker_model=None, remove_word_timing=False):
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.mean_dir_vec = mean_dir_vec
        self.remove_word_timing = remove_word_timing

        self.expected_audio_length = int(round(n_poses / pose_resampling_fps * 16000))
        self.expected_spectrogram_length = calc_spectrogram_length_from_motion_length(n_poses, pose_resampling_fps)

        self.lang_model = None

        print('Reading data \'{}\'...'.format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_s2eg_cache_new'
        if not os.path.exists(preloaded_dir):
            print('Creating the dataset cache...')
            assert mean_dir_vec is not None
            if mean_dir_vec.shape[-1] != 3:
                mean_dir_vec = mean_dir_vec.reshape(mean_dir_vec.shape[:-1] + (-1, 3))
            n_poses_extended = int(round(n_poses * 1.25))  # some margin
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses_extended,
                                            subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec)
            data_sampler.run()
        else:
            print('Found the cache {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

        # make a speaker model
        if speaker_model is None or speaker_model == 0:
            precomputed_model = lmdb_dir + '_s2eg_speaker_model.pkl'
            if not os.path.exists(precomputed_model):
                self._make_speaker_model(lmdb_dir, precomputed_model)
            else:
                with open(precomputed_model, 'rb') as f:
                    self.speaker_model = pickle.load(f)
        else:
            self.speaker_model = speaker_model
    
    def set_lang_model(self, lang_model):
        self.lang_model = lang_model

    def _make_speaker_model(self, lmdb_dir, cache_path):
        print('  building a speaker model...')
        speaker_model = Vocab('vid', insert_default_tokens=False)

        lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        txn = lmdb_env.begin(write=False)
        cursor = txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            speaker_model.index_word(vid)

        lmdb_env.close()
        print('    indexed %d videos' % speaker_model.n_words)
        self.speaker_model = speaker_model

        # cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.speaker_model, f)


def download_clips(vid_name, start_time, end_time, start_frame, end_frame, save_dir_vid, save_dir_wav):
    file_name = vid_name + '_' + str(start_frame) + '_' + str(end_frame)
    # wav_file = j(save_dir_wav, file_name + '.wav')
    # if not os.path.exists(wav_file):
    #     cmd_wav = ('ffmpeg $(youtube-dl -g \'https://www.youtube.com/watch?v={}\' |'
    #                ' sed \'s/.*/-ss {} -i &/\') -t {} -c:a copy {}')\
    #         .format(vid_name, video[-1]['start_time'],
    #                 video[-1]['end_time'] - video[-1]['start_time'], wav_file)
    #     return_code = os.system(cmd_wav)
    vid_file = j(save_dir_vid, file_name + '.mp4')
    wav_file = j(save_dir_wav, file_name + '.wav')
    # if vid_names_done[part_idx][key_idx] and not os.path.exists(vid_file):
    if not os.path.exists(vid_file):
        cmd_vid = ('ffmpeg -loglevel fatal $(youtube-dl -g \'https://www.youtube.com/watch?v={}\' |'
                   ' sed \'s/.*/-ss {} -i &/\') -t {} -c:v libx264 -c:a copy {}') \
            .format(vid_name, start_time, end_time - start_time, vid_file)
        return_code = os.system(cmd_vid)
        # if return_code != 0:
        #     vid_names_done[part_idx][key_idx] = False
    # if vid_names_done[part_idx][key_idx] and\
    #         os.path.exists(vid_file) and not os.path.exists(wav_file):
    if os.path.exists(vid_file) and not os.path.exists(wav_file):
        cmd_wav = 'ffmpeg -loglevel fatal -i {} -ac 2 -f wav {}'.format(vid_file, wav_file)
        os.system(cmd_wav)
    # print('\rPartition: {}. Key: {} of {} ({:.2f}%).'
    #       .format(partition, key_idx + 1, num_keys, 100. * (key_idx + 1) / num_keys), end='')


def load_ted_db_data(_path, dataset, config_args, ted_db_already_processed=False,
                     partition_data=False, block_size=300, filter_num=40):
    partitions = ['train', 'eval', 'test']
    vid_names_done = [[] for _ in range(len(partitions))]

    clip_duration_range = [5, 12]

    # load clips and make gestures
    mean_dir_vec = np.array(config_args.mean_dir_vec).reshape(-1, 3)
    train_dataset = TedDBParams(config_args.train_data_path[0],
                                n_poses=config_args.n_poses,
                                subdivision_stride=config_args.subdivision_stride,
                                pose_resampling_fps=config_args.motion_resampling_framerate,
                                mean_dir_vec=mean_dir_vec,
                                mean_pose=config_args.mean_pose,
                                remove_word_timing=(config_args.input_context == 'text')
                                )

    eval_dataset = TedDBParams(config_args.val_data_path[0],
                               n_poses=config_args.n_poses,
                               subdivision_stride=config_args.subdivision_stride,
                               pose_resampling_fps=config_args.motion_resampling_framerate,
                               mean_dir_vec=mean_dir_vec,
                               mean_pose=config_args.mean_pose,
                               remove_word_timing=(config_args.input_context == 'text')
                               )

    test_dataset = TedDBParams(config_args.test_data_path[0],
                               n_poses=config_args.n_poses,
                               subdivision_stride=config_args.subdivision_stride,
                               pose_resampling_fps=config_args.motion_resampling_framerate,
                               mean_dir_vec=mean_dir_vec,
                               mean_pose=config_args.mean_pose)

    # build vocab
    vocab_cache_path = j(os.path.split(config_args.train_data_path[0])[0],
                         'vocab_models_s2eg',
                         'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, eval_dataset, test_dataset],
                             vocab_cache_path, config_args.wordembed_path,
                             config_args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    eval_dataset.set_lang_model(lang_model)
    test_dataset.set_lang_model(lang_model)

    if not ted_db_already_processed:
        for part_idx, partition in enumerate(partitions):
            lmdb_env = lmdb.open(j(_path, dataset, 'lmdb_{}_s2eg_cache'.format(partition)),
                                 readonly=True, lock=False)
            save_dir_vid = j(_path, dataset, 'videos', partition)
            save_dir_wav = j(_path, dataset, 'waves', partition)
            os.makedirs(save_dir_vid, exist_ok=True)
            os.makedirs(save_dir_wav, exist_ok=True)
            with lmdb_env.begin(write=False) as txn:
                vid_names = []
                start_frames = []
                end_frames = []
                start_times = []
                end_times = []
                num_keys = 0
                # keys = [key for key, _ in txn.cursor()]
                # num_keys = len(keys)
                # vid_names = [''] * num_keys
                # start_frames = [0] * num_keys
                # end_frames = [0] * num_keys
                # start_times = [0.] * num_keys
                # end_times = [0.] * num_keys
                # for _key_idx, key in enumerate(keys):
                for key, _ in txn.cursor():
                    buf = txn.get(key)
                    video = pyarrow.deserialize(buf)
                    vid_names.append(video[-1]['vid'])
                    start_frames.append(video[-1]['start_frame_no'])
                    end_frames.append(video[-1]['end_frame_no'])
                    start_times.append(video[-1]['start_time'])
                    end_times.append(video[-1]['end_time'])
                    num_keys += 1
                    # vid_names[_key_idx] = video[-1]['vid']
                    # start_frames[_key_idx] = video[-1]['start_frame_no']
                    # end_frames[_key_idx] = video[-1]['end_frame_no']
                    # start_times[_key_idx] = video[-1]['start_time']
                    # end_times[_key_idx] = video[-1]['end_time']
                    # print('\rPartition: {}. Key {} of {}.'.format(partition, _key_idx + 1, num_keys), end='')
                    print('\rPartition: {}. Key {}.'.format(partition, num_keys), end='')
                # vid_names_done[part_idx] = True * np.ones(num_keys, dtype=bool)

                # for key_idx, key in enumerate(keys):
                # def download_clips(key):
                #     buf = txn.get(key)
                #     video = pyarrow.deserialize(buf)
                #     vid_name = video[-1]['vid']
                #     file_name = vid_name + '_' + str(video[-1]['start_frame_no']) +\
                #         '_' + str(video[-1]['end_frame_no'])
                #     # wav_file = j(save_dir_wav, file_name + '.wav')
                #     # if not os.path.exists(wav_file):
                #     #     cmd_wav = ('ffmpeg $(youtube-dl -g \'https://www.youtube.com/watch?v={}\' |'
                #     #                ' sed \'s/.*/-ss {} -i &/\') -t {} -c:a copy {}')\
                #     #         .format(vid_name, video[-1]['start_time'],
                #     #                 video[-1]['end_time'] - video[-1]['start_time'], wav_file)
                #     #     return_code = os.system(cmd_wav)
                #     vid_file = j(save_dir_vid, file_name + '.mp4')
                #     wav_file = j(save_dir_wav, file_name + '.wav')
                #     # if vid_names_done[part_idx][key_idx] and not os.path.exists(vid_file):
                #     if not os.path.exists(vid_file):
                #         cmd_vid = ('ffmpeg -loglevel fatal $(youtube-dl -g \'https://www.youtube.com/watch?v={}\' |'
                #                    ' sed \'s/.*/-ss {} -i &/\') -t {} -c:v libx264 -c:a copy {}')\
                #             .format(vid_name, video[-1]['start_time'],
                #                     video[-1]['end_time'] - video[-1]['start_time'], vid_file)
                #         return_code = os.system(cmd_vid)
                #         # if return_code != 0:
                #         #     vid_names_done[part_idx][key_idx] = False
                #     # if vid_names_done[part_idx][key_idx] and\
                #     #         os.path.exists(vid_file) and not os.path.exists(wav_file):
                #     if os.path.exists(vid_file) and not os.path.exists(wav_file):
                #         cmd_wav = 'ffmpeg -loglevel fatal -i {} -ac 2 -f wav {}'.format(vid_file, wav_file)
                #         os.system(cmd_wav)
                #     # print('\rPartition: {}. Key: {} of {} ({:.2f}%).'
                #     #       .format(partition, key_idx + 1, num_keys, 100. * (key_idx + 1) / num_keys), end='')

            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            pool.starmap(download_clips, [(vid_names[key_idx], start_times[key_idx], end_times[key_idx],
                                           start_frames[key_idx], end_frames[key_idx],
                                           save_dir_vid, save_dir_wav) for key_idx in tqdm(range(num_keys))])
    print('\nDownload complete.')

    if partition_data:
        lmdb_env = lmdb.open(j(_path, dataset, 'lmdb_train_s2eg_cache'), readonly=True, lock=False)
        map_size = 1024 * 50  # in MB
        map_size <<= 20  # in B
        lmdb_part_envs = [lmdb.open(j(_path, dataset, 'lmdb_train_s2eg_cache_new'), map_size=map_size),
                          lmdb.open(j(_path, dataset, 'lmdb_eval_s2eg_cache_new'), map_size=map_size),
                          lmdb.open(j(_path, dataset, 'lmdb_test_s2eg_cache_new'), map_size=map_size)]
        num_samples = [0, 0, 0]
        k_idx = 0
        with lmdb_env.begin(write=False) as txn:
            for key, _ in txn.cursor():
                buf = txn.get(key)
                video = pyarrow.deserialize(buf)
                vid_name = video[-1]['vid']
                start_frame = video[-1]['start_frame_no']
                end_frame = video[-1]['end_frame_no']
                for p_idx, partition in enumerate(partitions):
                    if os.path.exists(j(_path, dataset, 'waves', partition,
                                        '_'.join([vid_name, str(start_frame), str(end_frame)]) + '.wav')):
                        with lmdb_part_envs[p_idx].begin(write=True) as txn_part:
                            key_part = '{:010}'.format(num_samples[p_idx]).encode('ascii')
                            txn_part.put(key_part, buf)
                            num_samples[p_idx] += 1
                k_idx += 1
                print('\rProcessed keys: {}'.format(k_idx), end='')
        print()

    processed_dir = j(_path, dataset, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    data_wav_files = [j(processed_dir, 'train_data_wav.npz'),
                      j(processed_dir, 'eval_data_wav.npz'),
                      j(processed_dir, 'test_data_wav.npz')]
    # data_wav_dict_files = [j(processed_dir, 'train_data_wav_dict.npz'),
    #                        j(processed_dir, 'eval_data_wav_dict.npz'),
    #                        j(processed_dir, 'test_data_wav_dict.npz')]
    stats_file = j(processed_dir, 'stats.pkl')

    lmdb_part_envs = [lmdb.open(j(_path, dataset, 'lmdb_train_s2eg_cache_new'), readonly=True, lock=False),
                      lmdb.open(j(_path, dataset, 'lmdb_eval_s2eg_cache_new'), readonly=True, lock=False),
                      lmdb.open(j(_path, dataset, 'lmdb_test_s2eg_cache_new'), readonly=True, lock=False)]

    if not (os.path.exists(data_wav_files[0])
            and os.path.exists(data_wav_files[1])
            and os.path.exists(data_wav_files[2])
            and os.path.exists(stats_file)):
        max0 = None
        min0 = None
        max1 = None
        min1 = None
        max2 = None
        min2 = None
        for part_idx, partition in enumerate(partitions):
            save_dir_wav = j(_path, dataset, 'waves', partition)
            wav_files = glob.glob(j(save_dir_wav, '*.wav'))
            num_wav_files = len(wav_files)
            processed_dir = j(_path, dataset, 'processed', 'individual', partition)
            os.makedirs(processed_dir, exist_ok=True)
            # train_idx = np.random.choice(np.arange(num_wav_files), int(np.ceil(0.75 * num_wav_files)), replace=False)
            # eval_idx = np.random.choice(np.setdiff1d(np.arange(num_wav_files), train_idx),
            #                             int(np.ceil(0.15 * num_wav_files)), replace=False)
            # test_idx = np.setdiff1d(np.arange(num_wav_files), np.union1d(train_idx, eval_idx))
            # for wav_idx, wav_file_name in enumerate(wav_files):
            #     processed_data_file = j(_path, dataset, 'processed', 'individual',
            #                             partition, str(wav_idx).zfill(6) + '.npz')
            #     if wav_idx in eval_idx:
            #         os.rename(wav_file_name, j(_path, dataset, 'waves', 'eval', wav_file_name.split('/')[-1]))
            #         os.rename(processed_data_file,
            #                   j(_path, dataset, 'processed', 'individual', 'eval', str(wav_idx).zfill(6) + '.npz'))
            #     elif wav_idx in test_idx:
            #         os.rename(wav_file_name, j(_path, dataset, 'waves', 'test', wav_file_name.split('/')[-1]))
            #         os.rename(processed_data_file,
            #                   j(_path, dataset, 'processed', 'individual', 'test', str(wav_idx).zfill(6) + '.npz'))
            data_wav = np.zeros((num_wav_files, 3, block_size, filter_num), dtype=np.float16)
            # data_wav_names = [''] * num_wav_files
            bad_files = {'train': [], 'eval': [], 'test': []}
            with lmdb_part_envs[part_idx].begin(write=False) as txn_part:
                for key_byte, _ in txn_part.cursor():
                    sample = txn_part.get(key_byte)
                    key = int(key_byte.decode('ascii'))
                    sample = pyarrow.deserialize(sample)
                    vid_name = sample[-1]['vid']
                    clip_start = str(sample[-1]['start_frame_no'])
                    clip_end = str(sample[-1]['end_frame_no'])
                    wav_file_name = j(save_dir_wav, '_'.join([vid_name, clip_start, clip_end]) + '.wav')

                    processed_data_file = j(processed_dir, str(key).zfill(6) + '.npz')
                    # data_wav_names = '.'.join(wav_file_name.split('/')[-1].split('.')[:-1])
                    if os.path.exists(processed_data_file):
                        data_wav[key] = np.load(processed_data_file)['data']
                    else:
                        try:
                            data, time, rate = read_wav_file(wav_file_name)
                            mel_spec = ps.logfbank(data, rate, nfilt=filter_num, nfft=2048)
                            delta1 = ps.delta(mel_spec, 2)
                            delta2 = ps.delta(delta1, 2)

                            time = mel_spec.shape[0]
                            if time <= block_size:
                                part = mel_spec
                                delta11 = delta1
                                delta21 = delta2
                                part = np.pad(part, ((0, block_size - part.shape[0]), (0, 0)), 'constant',
                                              constant_values=0)
                                delta11 = np.pad(delta11, ((0, block_size - delta11.shape[0]), (0, 0)), 'constant',
                                                 constant_values=0)
                                delta21 = np.pad(delta21, ((0, block_size - delta21.shape[0]), (0, 0)), 'constant',
                                                 constant_values=0)
                                # train_data_1[train_num * block_size:(train_num + 1) * block_size] = part
                                # train_data_2[train_num * block_size:(train_num + 1) * block_size] = delta11
                                # train_data_3[train_num * block_size:(train_num + 1) * block_size] = delta21
                                data_wav[key, 0] = part
                                data_wav[key, 1] = delta11
                                data_wav[key, 2] = delta21
                            else:
                                for begin in np.arange(0, time, 100):
                                    end = begin + block_size
                                    end_from_last = time - begin
                                    begin_from_last = end_from_last - block_size
                                    if end > time:
                                        break

                                    part = mel_spec[begin:end, :]
                                    delta11 = delta1[begin:end, :]
                                    delta21 = delta2[begin:end, :]
                                    part_from_last = mel_spec[begin_from_last:end_from_last, :]
                                    delta11_from_last = delta1[begin_from_last:end_from_last, :]
                                    delta21_from_last = delta2[begin_from_last:end_from_last, :]

                                    data_wav[key, 0] = part
                                    data_wav[key, 1] = delta11
                                    data_wav[key, 2] = delta21

                                    # data_wav_list_1.append(part_from_last.tolist())
                                    # data_wav_list_2.append(delta11_from_last.tolist())
                                    # data_wav_list_3.append(delta21_from_last.tolist())
                                    # data_count += 2
                            np.savez_compressed(processed_data_file, data=data_wav[key])

                        except (wave.Error, EOFError):
                            bad_files[partition].append(wav_file_name)

                    print('\rPartition: {}. File: {} of {} ({:.2f}%).'
                          .format(partition, key + 1, num_wav_files,
                                  100. * (key + 1) / num_wav_files), end='')

            # mean1 = np.mean(train_data_wav_1, axis=(0, 1))
            # std1 = np.std(train_data_wav_1, axis=(0, 1))
            # mean2 = np.mean(train_data_wav_2, axis=(0, 1))
            # std2 = np.std(train_data_wav_2, axis=(0, 1))
            # mean3 = np.mean(train_data_wav_3, axis=(0, 1))
            # std3 = np.std(train_data_wav_3, axis=(0, 1))
            # train_data_wav = np.moveaxis(np.array([(train_data_wav_1 - mean1) / (std1 + epsilon),
            #                                        (train_data_wav_2 - mean2) / (std2 + epsilon),
            #                                        (train_data_wav_3 - mean3) / (std3 + epsilon)]),
            #                              0, 1)
            # eval_data_wav = np.moveaxis(np.array([(eval_data_wav_1 - mean1) / (std1 + epsilon),
            #                                       (eval_data_wav_2 - mean2) / (std2 + epsilon),
            #                                       (eval_data_wav_3 - mean3) / (std3 + epsilon)]),
            #                             0, 1)
            # test_data_wav = np.moveaxis(np.array([(test_data_wav_1 - mean1) / (std1 + epsilon),
            #                                       (test_data_wav_2 - mean2) / (std2 + epsilon),
            #                                       (test_data_wav_3 - mean3) / (std3 + epsilon)]),
            #                             0, 1)

            if part_idx == 0:
                max0 = np.max(data_wav[:, 0])
                min0 = np.min(data_wav[:, 0])
                max1 = np.max(data_wav[:, 1])
                min1 = np.min(data_wav[:, 1])
                max2 = np.max(data_wav[:, 2])
                min2 = np.min(data_wav[:, 2])

            data_wav[:, 0] = (data_wav[:, 0] - min0) / (max0 - min0)
            data_wav[:, 1] = (data_wav[:, 1] - min0) / (max0 - min0)
            data_wav[:, 2] = (data_wav[:, 2] - min0) / (max0 - min0)

            np.savez_compressed(data_wav_files[part_idx], data_wav)
            # np.savez_compressed(data_wav_dict_files[part_idx], data_wav_names)
            print('\nSuccessfully saved wave {} data.'.format(partition))

            if part_idx == 0:
                with open(stats_file, 'wb') as af:
                    pickle.dump((max0, min0, max1, min1, max2, min2), af)
                print('Successfully saved stats.')

    train_data_wav = np.load(data_wav_files[0])['arr_0']
    eval_data_wav = np.load(data_wav_files[1])['arr_0']
    test_data_wav = np.load(data_wav_files[2])['arr_0']
    # train_wav_dict = np.load(data_wav_dict_files[0], allow_pickle=True)['arr_0']
    # eval_wav_dict = np.load(data_wav_dict_files[1], allow_pickle=True)['arr_0']
    # test_wav_dict = np.load(data_wav_dict_files[2], allow_pickle=True)['arr_0']

    with open(stats_file, 'rb') as af:
        stats = pickle.load(af)
    max_all = np.array(stats[:3])
    min_all = np.array(stats[4:])

    # for part_idx, partition in enumerate(partitions):
    #     dir_wav = j(_path, dataset, 'waves', partition)
    #     wav_files = glob.glob(j(dir_wav, '*'))
    #     for wav_file in wav_files:
    #         audio_raw = librosa.load(wav_file, mono=True, sr=16000, res_type='kaiser_fast')

    return train_dataset, eval_dataset, test_dataset,\
        train_data_wav, eval_data_wav, test_data_wav,\
        max_all, min_all
        # train_wav_dict, eval_wav_dict, test_wav_dict,\


def build_vocab_idx(word_instants, min_word_count):
    # word to index dictionary
    word2idx = {
        constant.BOS_WORD: constant.BOS,
        constant.EOS_WORD: constant.EOS,
        constant.PAD_WORD: constant.PAD,
        constant.UNK_WORD: constant.UNK,
    }

    full_vocab = set(w for sent in word_instants for w in sent)
    print('Original Vocabulary size: {}'.format(len(full_vocab)))

    word_count = {w: 0 for w in full_vocab}

    # count word frequency in the given dataset
    for sent in word_instants:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)  # add word to dictionary with index
            else:
                ignored_word_count += 1

    print('Trimmed vocabulary size: {}\n\teach with minimum occurrence: {}'.format(len(word2idx), min_word_count))
    print('Ignored word count: {}'.format(ignored_word_count))

    return word2idx


def build_embedding_table(embedding_path, target_vocab):
    def load_emb_file(_embedding_path):
        vectors = []
        idx = 0
        _word2idx = dict()
        _idx2word = dict()
        with open(_embedding_path, 'r') as f:
            for l in tqdm(f):
                line = l.split()
                word = line[0]
                w_vec = np.array(line[1:]).astype(np.float)

                vectors.append(w_vec)
                _word2idx[word] = idx
                _idx2word[idx] = word
                idx += 1

        return np.array(vectors), _word2idx, _idx2word

    vectors, word2idx, idx2word = load_emb_file(embedding_path)
    dim = vectors.shape[1]

    embedding_table = np.zeros((len(target_vocab), dim))
    for k, v in target_vocab.items():
        try:
            embedding_table[v] = vectors[word2idx[k]]
        except KeyError:
            embedding_table[v] = np.random.normal(scale=0.6, size=(dim,))

    return embedding_table


def load_data_with_glove(_path, dataset, embedding_src, frame_drop=1, add_mirrored=False):
    data_path = j(_path, dataset)
    data_dict_file = j(data_path, 'data_dict_glove_drop_' + str(frame_drop) + '.npz')
    try:
        data_dict = np.load(data_dict_file, allow_pickle=True)['data_dict'].item()
        word2idx = np.load(data_dict_file, allow_pickle=True)['word2idx'].item()
        embedding_table = np.load(data_dict_file, allow_pickle=True)['embedding_table']
        tag_categories = list(np.load(data_dict_file, allow_pickle=True)['tag_categories'])
        max_time_steps = np.load(data_dict_file, allow_pickle=True)['max_time_steps'].item()
        print('Data file found. Returning data.')
    except FileNotFoundError:
        data_dict = []
        word2idx = []
        embedding_table = []
        tag_categories = []
        max_time_steps = 0.
        if dataset == 'mpi':
            channel_map = {
                'Xrotation': 'x',
                'Yrotation': 'y',
                'Zrotation': 'z'
            }
            data_dict = dict()
            tag_names = []
            with open(j(data_path, 'tag_names.txt')) as names_file:
                for line in names_file.readlines():
                    line = line[:-1]
                    tag_names.append(line)
            id = tag_names.index('ID')
            relevant_tags = ['Intended emotion', 'Intended polarity',
                             'Perceived category', 'Perceived polarity',
                             'Acting task', 'Gender', 'Age', 'Handedness', 'Native tongue', 'Text']
            tag_categories = [[] for _ in range(len(relevant_tags) - 1)]
            tag_files = glob.glob(j(data_path, 'tags/*.txt'))
            num_files = len(tag_files)
            for tag_file in tag_files:
                tag_data = []
                with open(tag_file) as f:
                    for line in f.readlines():
                        line = line[:-1]
                        tag_data.append(line)
                for category in range(len(tag_categories)):
                    tag_to_append = relevant_tags[category]
                    if tag_data[tag_names.index(tag_to_append)] not in tag_categories[category]:
                        tag_categories[category].append(tag_data[tag_names.index(tag_to_append)])

            all_texts = [[] for _ in range(len(tag_files))]
            for data_counter, tag_file in enumerate(tag_files):
                tag_data = []
                with open(tag_file) as f:
                    for line in f.readlines():
                        line = line[:-1]
                        tag_data.append(line)
                bvh_file = j(data_path, 'bvh/' + tag_data[id] + '.bvh')
                names, parents, offsets, \
                positions, rotations = MocapDataset.load_bvh(bvh_file, channel_map)
                positions_down_sampled = positions[1::frame_drop]
                rotations_down_sampled = rotations[1::frame_drop]
                if len(positions_down_sampled) > max_time_steps:
                    max_time_steps = len(positions_down_sampled)
                joints_dict = dict()
                joints_dict['joints_to_model'] = np.arange(len(parents))
                joints_dict['joints_parents_all'] = parents
                joints_dict['joints_parents'] = parents
                joints_dict['joints_names_all'] = names
                joints_dict['joints_names'] = names
                joints_dict['joints_offsets_all'] = offsets
                joints_dict['joints_left'] = [idx for idx, name in enumerate(names) if 'left' in name.lower()]
                joints_dict['joints_right'] = [idx for idx, name in enumerate(names) if 'right' in name.lower()]
                data_dict[tag_data[id]] = dict()
                data_dict[tag_data[id]]['joints_dict'] = joints_dict
                data_dict[tag_data[id]]['positions'] = positions_down_sampled
                data_dict[tag_data[id]]['rotations'] = rotations_down_sampled
                data_dict[tag_data[id]]['affective_features'] = \
                    MocapDataset.get_mpi_affective_features(positions_down_sampled)
                for tag_index, tag_name in enumerate(relevant_tags):
                    if tag_name.lower() == 'text':
                        all_texts[data_counter] = [e for e in str.split(tag_data[tag_names.index(tag_name)]) if
                                                   e.isalnum()]
                        data_dict[tag_data[id]][tag_name] = tag_data[tag_names.index(tag_name)]
                        text_length = len(data_dict[tag_data[id]][tag_name])
                        continue
                    if tag_name.lower() == 'age':
                        data_dict[tag_data[id]][tag_name] = float(tag_data[tag_names.index(tag_name)]) / 100.
                        continue
                    if tag_name is 'Perceived category':
                        categories = tag_categories[0]
                    elif tag_name is 'Perceived polarity':
                        categories = tag_categories[1]
                    else:
                        categories = tag_categories[tag_index]
                    data_dict[tag_data[id]][tag_name] = to_one_hot(tag_data[tag_names.index(tag_name)], categories)
                print('\rData file not found. Reading data files {}/{}: {:3.2f}%'.format(
                    data_counter + 1, num_files, data_counter * 100. / num_files), end='')
            print('\rData file not found. Reading files: done.')
            print('Preparing embedding table:')
            word2idx = build_vocab_idx(all_texts, min_word_count=0)
            embedding_table = build_embedding_table(embedding_src, word2idx)
            np.savez_compressed(data_dict_file,
                                data_dict=data_dict,
                                word2idx=word2idx,
                                embedding_table=embedding_table,
                                tag_categories=tag_categories,
                                max_time_steps=max_time_steps)
            print('done. Returning data.')
        elif dataset == 'creative_it':
            mocap_data_dirs = os.listdir(j(data_path, 'mocap'))
            for mocap_dir in mocap_data_dirs:
                mocap_data_files = glob.glob(j(data_path, 'mocap/' + mocap_dir + '/*.txt'))
        else:
            raise FileNotFoundError('Dataset not found.')

    return data_dict, word2idx, embedding_table, tag_categories, max_time_steps
