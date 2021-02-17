# sys
import glob
import lmdb
import numpy as np
import os
import pickle
import pyarrow
import python_speech_features as ps
import pyttsx3
import re
import wave

import utils.constant as constant

from nltk.stem.porter import PorterStemmer
from os.path import join as j
from scipy.io import wavfile
from tqdm import tqdm

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


def extract_dimensional_emotions(string):
    # a: dimensional emotion, c: categorical emotion
    # e: evaluator, f/m: self-reported
    if string[:3].lower() == 'a-e':
        emotions = string.split()
        emotions = [0. if emotions[i] == ';'
                    else float(emotions[i].split(';')[0])
                    for i in [2, 4, 6]]
        return emotions
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
    processed_dir = j(dataset_dir, 'processed_new')
    os.makedirs(processed_dir, exist_ok=True)
    train_data_wav_file = j(processed_dir, 'train_data_wav.npz')
    eval_data_wav_file = j(processed_dir, 'eval_data_wav.npz')
    test_data_wav_file = j(processed_dir, 'test_data_wav.npz')
    train_labels_file = j(processed_dir, 'train_labels.npz')
    eval_labels_file = j(processed_dir, 'eval_labels.npz')
    test_labels_file = j(processed_dir, 'test_labels.npz')
    stats_file = j(processed_dir, 'stats.pkl')

    if not (os.path.exists(train_data_wav_file)
            and os.path.exists(eval_data_wav_file)
            and os.path.exists(test_data_wav_file)
            and os.path.exists(train_data_wav_file)
            and os.path.exists(eval_data_wav_file)
            and os.path.exists(test_data_wav_file)
            and os.path.exists(stats_file)):

        session_set_train = [1, 2, 3, 4]
        session_set_test = [5]
        data_wav_list_1 = []
        data_wav_list_2 = []
        data_wav_list_3 = []
        labels_list = []
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
                emotions = []
                with open(emo_file, 'r') as ef:
                    ef_lines = ef.readlines()
                    for ef_line in ef_lines:
                        if ef_line[0] == '[':
                            emotions.append([float(x) for x in re.findall('\d+\.\d+', ef_line)[-3:]])

                        # extract_dimensional_emotions(ef_line)

                wav_files = glob.glob(j(wav_dir, sess, '*.wav'))
                num_wav_files = len(wav_files)
                assert num_wav_files == len(emotions), 'Number of annotations do not match number of .wav files'
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

                        labels_list.append(emotions[wav_idx])

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

                            labels_list.append(emotions[wav_idx])

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

        train_labels = \
            np.array([labels_list[i] for i in train_idx]) - (dimensional_max - dimensional_min) / 2.
        eval_labels = \
            np.array([labels_list[i] for i in eval_idx]) - (dimensional_max - dimensional_min) / 2.
        test_labels = \
            np.array([labels_list[i] for i in test_idx]) - (dimensional_max - dimensional_min) / 2.

        mean1 = np.mean(train_data_wav_1, axis=(0, 1))
        std1 = np.std(train_data_wav_1, axis=(0, 1))
        mean2 = np.mean(train_data_wav_2, axis=(0, 1))
        std2 = np.std(train_data_wav_2, axis=(0, 1))
        mean3 = np.mean(train_data_wav_3, axis=(0, 1))
        std3 = np.std(train_data_wav_3, axis=(0, 1))

        train_data_wav = np.moveaxis(np.array([(train_data_wav_1 - mean1) / (std1 + epsilon),
                                               (train_data_wav_2 - mean2) / (std2 + epsilon),
                                               (train_data_wav_3 - mean3) / (std3 + epsilon)]),
                                     0, 1)
        eval_data_wav = np.moveaxis(np.array([(eval_data_wav_1 - mean1) / (std1 + epsilon),
                                              (eval_data_wav_2 - mean2) / (std2 + epsilon),
                                              (eval_data_wav_3 - mean3) / (std3 + epsilon)]),
                                    0, 1)
        test_data_wav = np.moveaxis(np.array([(test_data_wav_1 - mean1) / (std1 + epsilon),
                                              (test_data_wav_2 - mean2) / (std2 + epsilon),
                                              (test_data_wav_3 - mean3) / (std3 + epsilon)]),
                                    0, 1)

        np.savez_compressed(train_data_wav_file, train_data_wav)
        print('Successfully saved wave train data.')
        np.savez_compressed(eval_data_wav_file, eval_data_wav)
        print('Successfully saved wave eval data.')
        np.savez_compressed(test_data_wav_file, test_data_wav)
        print('Successfully saved wave test data.')

        np.savez_compressed(train_labels_file, train_labels)
        print('Successfully saved train labels.')
        np.savez_compressed(eval_labels_file, eval_labels)
        print('Successfully saved eval labels.')
        np.savez_compressed(test_labels_file, test_labels)
        print('Successfully saved test labels.')

        with open(stats_file, 'wb') as af:
            pickle.dump((mean1, std1, mean2, std2, mean3, std3), af)
        means = np.array([mean1, mean2, mean3])
        stds = np.array([std1, std2, std3])
        print('Successfully saved stats.')
    else:
        train_data_wav = np.load(train_data_wav_file)['arr_0']
        eval_data_wav = np.load(eval_data_wav_file)['arr_0']
        test_data_wav = np.load(test_data_wav_file)['arr_0']

        train_labels = np.load(train_labels_file)['arr_0']
        eval_labels = np.load(eval_labels_file)['arr_0']
        test_labels = np.load(test_labels_file)['arr_0']

        with open(stats_file, 'rb') as af:
            stats = pickle.load(af)
        means = np.array(stats[:3])
        stds = np.array(stats[4:])

    return train_data_wav, eval_data_wav, test_data_wav, \
           train_labels, eval_labels, test_labels, \
           means, stds


def load_ted_db_data(_path, dataset):
    partitions = ['train', 'val', 'test']

    clip_duration_range = [5, 12]

    # load clips and make gestures
    n_saved = 0
    for partition in partitions:
        lmdb_env = lmdb.open(j(_path, dataset, 'lmdb_{}'.format(partition)), readonly=True, lock=False)
        save_dir_vid = j(_path, dataset, 'videos', partition)
        save_dir_wav = j(_path, dataset, 'waves', partition)
        os.makedirs(save_dir_vid, exist_ok=True)
        os.makedirs(save_dir_wav, exist_ok=True)
        vid_names = []
        num_clips = []
        clip_start = []
        clip_end = []
        with lmdb_env.begin(write=False) as txn:
            keys = [key for key, _ in txn.cursor()]
            num_videos = len(keys)
            for key_idx, key in enumerate(keys):
                buf = txn.get(key)
                video = pyarrow.deserialize(buf)
                vid_name = video['vid']
                clips = video['clips']
                num_clips = len(clips)
                for clip_idx, clip in enumerate(clips):
                    clip_start.append(clip['start_time'])
                    clip_end.append(clip['end_time'])
                    file_name = vid_name + '_' + str(clip_idx).zfill(3)
                    vid_file = j(save_dir_vid, file_name + '.mp4')
                    wav_file = j(save_dir_wav, file_name + '.wav')
                    if not os.path.exists(vid_file):
                        cmd_vid = ('ffmpeg $(youtube-dl -g \'https://www.youtube.com/watch?v={}\' |'
                                   ' sed "s/.*/-ss {} -i &/") -t {} -c:v libx264 -c:a copy {}')\
                            .format(vid_name, clip['start_time'], clip['end_time'] - clip['start_time'], vid_file)
                        os.system(cmd_vid)
                    if not os.path.exists(wav_file):
                        cmd_wav = 'ffmpeg -i {} -ac 2 -f wav {}'.format(vid_file, wav_file)
                        os.system(cmd_wav)
                    print('\rPartition: {}. Video: {} of {} ({:.2f}%). Clip: {} of {} ({:.2f}%).'
                          .format(partition, key_idx + 1, num_videos, 100. * (key_idx + 1) / num_videos,
                                  clip_idx + 1, num_clips, 100. * (clip_idx + 1) / num_clips), end='')
    print()
    num_clips = np.array(num_clips)
    clip_start = np.array(clip_start)
    clip_end = np.array(clip_end)
    clip_lengths = clip_end - clip_start
    return True


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

            all_texts = [[]] * len(tag_files)
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
