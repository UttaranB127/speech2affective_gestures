import torch
import torch.nn as nn

import net.embedding_net as en
import utils.ted_db_utils as ted_db
# import net.embedding_net as embedding_net

from net.tcn import TemporalConvNet
from net.utils.graph import Graph
from net.utils.tgcn import STGraphConv, STGraphConvTranspose
from utils import vocab


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


class MFCCEncoder(nn.Module):
    def __init__(self, mfcc_length, num_mfcc, time_steps):
        super().__init__()
        self.conv1 = nn.Conv1d(mfcc_length, 64, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 48, 3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(48)
        self.conv4 = nn.Conv1d(48, time_steps, 3, padding=1)
        self.batch_norm4 = nn.BatchNorm1d(time_steps)

        self.linear1 = nn.Linear(num_mfcc, 32)

        self.activation = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, mfcc_data):
        x_01 = self.activation(self.batch_norm1(self.conv1(mfcc_data.permute(0, 2, 1))))
        x_02 = self.activation(self.batch_norm2(self.conv2(x_01)))
        x_03 = self.activation(self.batch_norm3(self.conv3(x_02)))
        x_04 = self.activation(self.batch_norm4(self.conv4(x_03)))
        out = self.activation(self.linear1(x_04))
        return out


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

    def forward(self, in_data):
        emb = self.drop(self.embedding(in_data))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0


class AffEncoder(nn.Module):
    def __init__(self, coords=3):
        super().__init__()
        self.coords = coords

        graph1 = Graph(len(ted_db.dir_edge_pairs) + 1,
                       ted_db.dir_edge_pairs,
                       strategy='spatial',
                       max_hop=2)
        self.A1 = torch.tensor(graph1.A,
                               dtype=torch.float32,
                               requires_grad=False).cuda()

        self.num_body_parts = len(ted_db.body_parts_edge_idx)
        graph2 = Graph(len(ted_db.body_parts_edge_pairs) + 1,
                       ted_db.body_parts_edge_pairs,
                       strategy='spatial',
                       max_hop=2)
        self.A2 = torch.tensor(graph2.A,
                               dtype=torch.float32,
                               requires_grad=False).cuda()

        spatial_kernel_size1 = 5
        temporal_kernel_size1 = 9
        kernel_size1 = (temporal_kernel_size1, spatial_kernel_size1)
        padding1 = ((kernel_size1[0] - 1) // 2, (kernel_size1[1] - 1) // 2)
        self.st_gcn1 = STGraphConv(coords, 16, self.A1.size(0), kernel_size1,
                                   stride=(1, 1), padding=padding1)

        spatial_kernel_size2 = 3
        temporal_kernel_size2 = 9
        kernel_size2 = (temporal_kernel_size2, spatial_kernel_size2)
        padding2 = ((kernel_size2[0] - 1) // 2, (kernel_size2[1] - 1) // 2)
        self.st_gcn2 = STGraphConv(48, 16, self.A2.size(0), kernel_size2,
                                   stride=(1, 1), padding=padding2)
        # self.pre_conv = nn.Sequential(
        #     nn.Conv1d(input_size, 16, 3),
        #     nn.BatchNorm1d(16),
        #     nn.LeakyReLU(True),
        #     nn.Conv1d(16, 8, 3),
        #     nn.BatchNorm1d(8),
        #     nn.LeakyReLU(True),
        #     nn.Conv1d(8, 8, 3),
        # )
        kernel_size3 = 5
        padding3 = (kernel_size3 - 1) // 2
        self.conv1 = nn.Conv1d(48, 16, kernel_size3, padding=padding3)
        self.batch_norm1 = nn.BatchNorm1d(16)

        kernel_size4 = 3
        padding4 = (kernel_size4 - 1) // 2
        self.conv2 = nn.Conv1d(16, 8, kernel_size4, padding=padding4)
        self.batch_norm2 = nn.BatchNorm1d(8)

        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, poses):
        n, t, jc = poses.shape
        j = jc // self.coords
        # poses = torch.cat((poses, torch.zeros_like(poses[..., 0:3])), dim=-1)
        feat1_out, _ = self.st_gcn1(poses.view(n, t, -1, 3).permute(0, 3, 1, 2), self.A1)
        f1 = feat1_out.shape[1]
        feat2_in = torch.zeros((n, t,
                                ted_db.max_body_part_edges * f1,
                                self.num_body_parts)).float().cuda()
        for idx, body_part_idx in enumerate(ted_db.body_parts_edge_idx):
            feat2_in[..., :f1 * len(body_part_idx), idx] =\
                feat1_out[..., body_part_idx].permute(0, 2, 1, 3).contiguous().view(n, t, -1)
        feat2_in = feat2_in.permute(0, 2, 1, 3)
        feat2_out, _ = self.st_gcn2(feat2_in, self.A2)
        feat3_in = feat2_out.permute(0, 2, 1, 3).contiguous().view(n, t, -1).permute(0, 2, 1)
        feat3_out = self.activation(self.batch_norm1(self.conv1(feat3_in)))
        feat4_out = self.activation(self.batch_norm2(self.conv2(feat3_out))).permute(0, 2, 1)

        return feat4_out


class AffDecoder(nn.Module):
    def __init__(self, coords=3, num_joints=9):
        super().__init__()
        self.coords = coords
        self.num_joints = num_joints

        # kernel_size1 = 3
        # padding1 = (kernel_size1 - 1) // 2
        # self.conv1 = nn.ConvTranspose1d(8, 16, kernel_size1, padding=padding1)
        # self.batch_norm1 = nn.BatchNorm1d(16)
        #
        # kernel_size2 = 5
        # padding2 = (kernel_size2 - 1) // 2
        # self.conv2 = nn.ConvTranspose1d(16, 48, kernel_size2, padding=padding2)
        # self.batch_norm2 = nn.BatchNorm1d(48)

        self.num_body_parts = len(ted_db.body_parts_edge_idx)
        graph1 = Graph(len(ted_db.body_parts_edge_pairs) + 1,
                       ted_db.body_parts_edge_pairs,
                       strategy='spatial',
                       max_hop=2)
        self.A1 = torch.tensor(graph1.A,
                               dtype=torch.float32,
                               requires_grad=False).cuda()

        graph2 = Graph(len(ted_db.dir_edge_pairs) + 1,
                       ted_db.dir_edge_pairs,
                       strategy='spatial',
                       max_hop=2)
        self.A2 = torch.tensor(graph2.A,
                               dtype=torch.float32,
                               requires_grad=False).cuda()

        spatial_kernel_size1 = 3
        temporal_kernel_size1 = 9
        kernel_size1 = (temporal_kernel_size1, spatial_kernel_size1)
        padding1 = ((kernel_size1[0] - 1) // 2, (kernel_size1[1] - 1) // 2)
        self.st_gcn1 = STGraphConvTranspose(16, 48, self.A1.size(0), kernel_size1,
                                            stride=(1, 1), padding=padding1)

        spatial_kernel_size2 = 5
        temporal_kernel_size2 = 9
        kernel_size2 = (temporal_kernel_size2, spatial_kernel_size2)
        padding2 = ((kernel_size2[0] - 1) // 2, (kernel_size2[1] - 1) // 2)
        self.st_gcn2 = STGraphConvTranspose(16, self.coords, self.A2.size(0), kernel_size2,
                                            stride=(1, 1), padding=padding2)

        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, pose_feats):
        n, t, f = pose_feats.shape

        # feat1_out = self.activation(self.batch_norm1(self.conv1(pose_feats.permute(0, 2, 1))))
        # feat2_out = self.activation(self.batch_norm2(self.conv2(feat1_out)))
        # feat3_in = feat2_out.permute(0, 2, 1).contiguous().view(n, t, self.num_body_parts, -1).permute(0, 3, 1, 2)

        # feat3_in = pose_feats.contiguous().view(n, t, self.num_body_parts, -1).permute(0, 3, 1, 2)
        # feat3_out, _ = self.st_gcn1(feat3_in, self.A1)
        # feat4_out, _ = self.st_gcn2(feat3_out.permute(0, 2, 1, 3).contiguous().
        #                             view(n, t, -1, self.num_joints).permute(0, 2, 1, 3),
        #                             self.A2)

        feat4_in = pose_feats.contiguous().view(n, t, self.num_joints, -1).permute(0, 3, 1, 2)
        feat4_out, _ = self.st_gcn2(feat4_in,
                                    self.A2)

        return feat4_out.permute(0, 2, 3, 1).contiguous().view(n, t, -1)


class PoseGeneratorTriModal(nn.Module):
    def __init__(self, args, pose_dim, n_words, word_embed_size, word_embeddings, z_obj=None):
        super().__init__()
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.z_obj = z_obj
        self.input_context = args.input_context

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
            self.in_size += self.z_size
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

    def forward(self, pre_seq, in_text, in_audio, vid_indices=None):
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
                assert vid_indices is not None
                z_context = self.speaker_embedding(vid_indices)
                z_mu = self.speaker_mu(z_context)
                z_log_var = self.speaker_log_var(z_context)
                z_context = en.re_parametrize(z_mu, z_log_var)
            else:
                z_mu = z_log_var = None
                z_context = torch.randn(in_text.shape[0], self.z_size, device=in_text.device)
        else:
            z_mu = z_log_var = None
            z_context = None

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


class DiscriminatorTriModal(nn.Module):
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


class ConvDiscriminatorTriModal(nn.Module):
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


class PoseGenerator(nn.Module):
    def __init__(self, args, pose_dim, n_words, word_embed_size, word_embeddings,
                 mfcc_length, num_mfcc, time_steps, z_obj=None):
        super().__init__()
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.z_obj = z_obj
        self.input_context = args.input_context
        self.mfcc_feature_length = 32
        self.text_feature_length = 32
        self.pose_feature_length = 8

        if self.input_context == 'both':
            # audio_feat + text_feat + last pose + constraint bit
            self.in_size = self.mfcc_feature_length + self.text_feature_length + self.pose_feature_length
        elif self.input_context == 'audio':
            self.in_size = self.mfcc_feature_length + self.pose_feature_length  # audio only
        elif self.input_context == 'text':
            self.in_size = self.text_feature_length + self.pose_feature_length  # text only
        elif self.input_context == 'none':
            self.in_size = self.pose_feature_length

        self.audio_encoder = MFCCEncoder(mfcc_length, num_mfcc, time_steps)
        self.text_encoder = TextEncoderTCN(args, n_words, word_embed_size, pre_trained_embedding=word_embeddings,
                                           dropout=args.dropout_prob)
        self.aff_encoder = AffEncoder()

        self.speaker_embedding = None
        if self.z_obj:
            self.z_size = 16
            self.in_size += self.z_size
            if self.z_obj.__class__.__name__ == 'Vocab':
                self.speaker_embedding = nn.Sequential(
                    nn.Embedding(z_obj.n_words, self.z_size),
                    nn.Linear(self.z_size, self.z_size)
                )
                self.speaker_mu = nn.Linear(self.z_size, self.z_size)
                self.speaker_log_var = nn.Linear(self.z_size, self.z_size)
            else:
                pass  # random noise

        self.hidden_size = args.hidden_size_s2eg
        self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=args.n_layers, batch_first=True,
                          bidirectional=True, dropout=args.dropout_prob)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size//2, pose_dim)
        )

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, pre_seq, in_text, in_mfcc, vid_indices=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        text_feat_seq = audio_feat_seq = None
        if self.input_context != 'none':
            # audio
            audio_feat_seq = self.audio_encoder(in_mfcc)  # output (bs, n_frames, feat_size)

            # text
            text_feat_seq, _ = self.text_encoder(in_text)
            assert(audio_feat_seq.shape[1] == text_feat_seq.shape[1]),\
                'Audio and text features must have the same number of time steps. ' \
                'Found time steps: audio features: {}, text features: {}.'.format(audio_feat_seq.shape[1],
                                                                                  text_feat_seq.shape[1])

        # z vector; speaker embedding or random noise
        if self.z_obj:
            if self.speaker_embedding:
                assert vid_indices is not None
                z_context = self.speaker_embedding(vid_indices)
                z_mu = self.speaker_mu(z_context)
                z_log_var = self.speaker_log_var(z_context)
                z_context = en.re_parametrize(z_mu, z_log_var)
            else:
                z_mu = z_log_var = None
                z_context = torch.randn(in_text.shape[0], self.z_size, device=in_text.device)
        else:
            z_mu = z_log_var = None
            z_context = None

        pre_feat_seq = self.aff_encoder(pre_seq[..., :-1])
        if self.input_context == 'both':
            in_data = torch.cat((pre_feat_seq, audio_feat_seq, text_feat_seq), dim=2)
        elif self.input_context == 'audio':
            in_data = torch.cat((pre_feat_seq, audio_feat_seq), dim=2)
        elif self.input_context == 'text':
            in_data = torch.cat((pre_feat_seq, text_feat_seq), dim=2)
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


class AffDiscriminator(nn.Module):
    def __init__(self, input_size, coords=3):
        super().__init__()
        self.input_size = input_size
        self.coords = coords
        self.hidden_size = 64

        self.aff_encoder = AffEncoder(coords=coords)

        self.gru = nn.GRU(8, hidden_size=self.hidden_size,
                          num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(34, 1)

        self.activation = nn.LeakyReLU(inplace=True)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        n, t, jc = poses.shape
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        aff_encoder_out = self.aff_encoder(poses)
        output_gru, decoder_hidden = self.gru(aff_encoder_out, decoder_hidden)
        # sum bidirectional outputs
        output_gru_bi = output_gru[:, :, :self.hidden_size] + output_gru[:, :, self.hidden_size:]

        # apply linear to every output
        output_linear1 = self.out(output_gru_bi.contiguous().view(-1, output_gru_bi.shape[2])).view(n, -1)
        output_linear2 = self.out2(output_linear1)

        return torch.sigmoid(output_linear2)
