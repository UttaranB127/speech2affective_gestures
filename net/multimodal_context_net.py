import torch
import torch.nn as nn

# import net.embedding_net as embedding_net

from utils import vocab
from net.tcn import TemporalConvNet


class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6),
        )

    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)


class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layers
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], 32)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0


class PoseGenerator(nn.Module):
    def __init__(self, args, pose_dim, n_words, word_embed_size, word_embeddings, labels_size, z_obj=None):
        super().__init__()
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.z_obj = z_obj
        self.input_context = args.input_context
        self.labels_size = labels_size

        if self.input_context == 'both':
            self.in_size = 32 + 32 + pose_dim + 1  # audio_feat + text_feat + last pose + constraint bit
        elif self.input_context == 'none':
            self.in_size = pose_dim + 1
        else:
            self.in_size = 32 + pose_dim + 1  # audio or text only

        self.audio_encoder = WavEncoder()
        self.text_encoder = TextEncoderTCN(args, n_words, word_embed_size, pre_trained_embedding=word_embeddings,
                                           dropout=args.dropout_prob)

        self.speaker_embedding = None
        if self.z_obj:
            self.z_size = 16
            self.in_size += (self.z_size + self.labels_size)
            if self.z_obj.__class__.__name__ == 'Vocab':
                self.speaker_embedding = nn.Sequential(
                    nn.Embedding(z_obj.n_words, self.z_size),
                    nn.Linear(self.z_size, self.z_size)
                )
                self.speaker_mu = nn.Linear(self.z_size, self.z_size)
                self.speaker_log_var = nn.Linear(self.z_size, self.z_size)
            else:
                pass  # random noise

        self.hidden_size = args.hidden_size
        self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=args.n_layers, batch_first=True,
                          bidirectional=True, dropout=args.dropout_prob)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size//2, pose_dim)
        )

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    @staticmethod
    def re_parametrize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pre_seq, in_text, in_audio, in_emo_labels, vid_indices=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        text_feat_seq = audio_feat_seq = None
        if self.input_context != 'none':
            # audio
            audio_feat_seq = self.audio_encoder(in_audio)  # output (bs, n_frames, feat_size)

            # text
            text_feat_seq, _ = self.text_encoder(in_text)
            assert(audio_feat_seq.shape[1] == text_feat_seq.shape[1])

        # z vector; speaker embedding or random noise
        if self.z_obj:
            if self.speaker_embedding:
                try:
                    assert vid_indices is not None
                except AssertionError:
                    temp = 1
                z_context = self.speaker_embedding(vid_indices)
                z_mu = self.speaker_mu(z_context)
                z_log_var = self.speaker_log_var(z_context)
                z_context = PoseGenerator.re_parametrize(z_mu, z_log_var)
            else:
                z_mu = z_log_var = None
                z_context = torch.randn(in_text.shape[0], self.z_size, device=in_text.device)
        else:
            z_mu = z_log_var = None
            z_context = None

        # emotion
        z_context = torch.cat((z_context, in_emo_labels), dim=-1)

        if self.input_context == 'both':
            in_data = torch.cat((pre_seq, audio_feat_seq, text_feat_seq), dim=2)
        elif self.input_context == 'audio':
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        elif self.input_context == 'text':
            in_data = torch.cat((pre_seq, text_feat_seq), dim=2)
        elif self.input_context == 'none':
            in_data = pre_seq
        else:
            assert False

        if z_context is not None:
            repeated_z = z_context.unsqueeze(1)
            repeated_z = repeated_z.repeat(1, in_data.shape[1], 1)
            in_data = torch.cat((in_data, repeated_z), dim=2)

        output, decoder_hidden = self.gru(in_data, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs
        output = self.out(output.reshape(-1, output.shape[2]))
        decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)

        return decoder_outputs, z_context, z_mu, z_log_var


class Discriminator(nn.Module):
    def __init__(self, args, input_size, n_words=None, word_embed_size=None, word_embeddings=None):
        super().__init__()
        self.input_size = input_size

        if n_words and word_embed_size:
            self.text_encoder = TextEncoderTCN(n_words, word_embed_size, word_embeddings)
            input_size += 32
        else:
            self.text_encoder = None

        self.hidden_size = args.hidden_size
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_size, num_layers=args.n_layers, bidirectional=True,
                          dropout=args.dropout_prob, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(args.n_poses, 1)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        if self.text_encoder:
            text_feat_seq, _ = self.text_encoder(in_text)
            poses = torch.cat((poses, text_feat_seq), dim=2)

        output, decoder_hidden = self.gru(poses, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs

        # use the last N outputs
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)

        return output


class ConvDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True),
            nn.Conv1d(8, 8, 3),
        )

        self.gru = nn.GRU(8, hidden_size=self.hidden_size, num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(28, 1)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        poses = poses.transpose(1, 2)
        feat = self.pre_conv(poses)
        feat = feat.transpose(1, 2)

        output, decoder_hidden = self.gru(feat, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs

        # use the last N outputs
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)

        return output
