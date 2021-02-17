from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from utils.common import *
from utils.Quaternions_torch import qmul, qeuler, euler_to_quaternion

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

torch.manual_seed(1234)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class T2GNet(nn.Module):

    def __init__(self, num_tokens, embedding_table, max_time_steps, text_dim, quat_dim, quat_channels,
                 offsets_dim, intended_emotion_dim, intended_polarity_dim, acting_task_dim,
                 gender_dim, age_dim, handedness_dim, native_tongue_dim, num_heads, num_hidden_units,
                 num_layers, dropout=0.5):
        super(T2GNet, self).__init__()
        self.T = max_time_steps
        self.text_dim = text_dim
        self.quat_channels = quat_channels
        self.text_mask = None
        self.quat_mask = None
        self.text_embedding = nn.Embedding.from_pretrained(embedding_table, freeze=True)
        self.text_pos_encoder = PositionalEncoding(text_dim, dropout)
        encoder_layers = TransformerEncoderLayer(text_dim, num_heads, num_hidden_units, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        intermediate_dim = int((text_dim + quat_dim) / 2)
        self.text_embed = nn.Linear(text_dim + intended_emotion_dim + intended_polarity_dim +\
            acting_task_dim + gender_dim + age_dim + handedness_dim + native_tongue_dim, intermediate_dim)
        self.text_offsets_to_gestures = nn.Linear(intermediate_dim + offsets_dim, quat_dim)

        self.quat_pos_encoder = PositionalEncoding(quat_dim, dropout)
        decoder_layers = TransformerDecoderLayer(quat_dim, num_heads, num_hidden_units, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        self.temporal_smoothing = nn.ModuleList((
            nn.Conv1d(max_time_steps, max_time_steps, 3, padding=1),
            nn.Conv1d(max_time_steps, max_time_steps, 3, padding=1),
        ))
        self.decoder = nn.Linear(text_dim, num_tokens)

        self.init_weights()

    # def __init__(self, V, D, S, A, O, Z, RS, L,
        #              **kwargs):
        #     super().__init__()
        #
        # self.V = V
        # self.D = D
        # self.S = S
        # self.A = A
        # self.O = O
        # self.Z = Z
        # self.RS = RS
        # self.L = L
        #
        # geom_fc1_size = 16
        # geom_fc2_size = 16
        # geom_fc3_size = 8
        #
        # self.geom_fc1 = nn.Linear(A + L, geom_fc1_size)
        # self.geom_fc2 = nn.Linear(geom_fc1_size, geom_fc2_size)
        # self.geom_fc3 = nn.Linear(geom_fc2_size, geom_fc3_size)
        #
        # spline_fc1_size = 8
        # spline_fc2_size = 4
        # self.spline_fc1 = nn.Linear(S + L, spline_fc1_size)
        # self.spline_fc2 = nn.Linear(spline_fc1_size, spline_fc2_size)
        #
        # self.relu = nn.ReLU(inplace=True)
        # self.lrelu = nn.LeakyReLU(0.05, inplace=True)
        # self.elu = nn.ELU(0.05, inplace=True)
        #
        # quat_h_size = 1000
        # self.quat_rnn = nn.GRU(input_size=(V - 1) * D + geom_fc3_size + spline_fc2_size + self.O + self.Z + self.RS,
        #                        hidden_size=quat_h_size, num_layers=2,
        #                        batch_first=True)
        # self.quat_h0 = nn.Parameter(torch.zeros(self.quat_rnn.num_layers, 1, quat_h_size).normal_(std=0.01),
        #                             requires_grad=True)
        #
        # quat_fc4_size = 128
        # self.quat_fc4 = nn.Linear(quat_h_size, quat_fc4_size)
        # self.quat_fc5 = nn.Linear(quat_fc4_size, (V - 1) * D)
        #
        # o_z_rs_fc4_size = 4
        # self.o_z_rs_fc4 = nn.Linear(O + Z + RS + geom_fc3_size + spline_fc2_size, o_z_rs_fc4_size)
        # self.o_fc5 = nn.Linear(o_z_rs_fc4_size, O)
        # self.z_fc5 = nn.Linear(o_z_rs_fc4_size, Z)
        # self.rs_fc5 = nn.Linear(o_z_rs_fc4_size, RS)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.text_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, quat, o_z_rs, affs, spline, labels, quat_h=None,
    #             return_prenorm=False, return_all=False, teacher_steps=0):
    #
    #     geom_controls = torch.cat((affs, labels), dim=-1)
    #     geom_controls = self.elu(self.geom_fc1(geom_controls))
    #     geom_controls = self.elu(self.geom_fc2(geom_controls))
    #     geom_controls = self.elu(self.geom_fc3(geom_controls))
    #
    #     spline_controls = torch.cat((spline, labels), dim=-1)
    #     spline_controls = self.elu(self.spline_fc1(spline_controls))
    #     spline_controls = self.elu(self.spline_fc2(spline_controls))
    #
    #     o_z_rs_combined = torch.cat((o_z_rs, geom_controls, spline_controls), dim=-1)
    #
    #     quat_combined = torch.cat((quat[:, :, :self.V * self.D],
    #                                geom_controls, spline_controls, o_z_rs), dim=-1)
    #
    #     if quat_h is None:
    #         quat_h = self.quat_h0.expand(-1, quat.shape[0], -1).contiguous()
    #     quat_combined, quat_h = self.quat_rnn(quat_combined, quat_h)
    #
    #     quat = self.elu(self.quat_fc4(quat_combined))
    #
    #     o_z_rs = self.lrelu(self.o_z_rs_fc4(o_z_rs_combined))
    #
    #     if return_all:
    #         quat = self.quat_fc5(quat)
    #         o = self.o_fc5(o_z_rs)
    #         z = self.z_fc5(o_z_rs)
    #         rs = torch.abs(self.rs_fc5(o_z_rs))
    #     else:
    #         quat = self.quat_fc5(quat[:, -1:])
    #         o = self.o_fc5(o_z_rs[:, -1:])
    #         z = self.z_fc5(o_z_rs[:, -1:])
    #         rs = torch.abs(self.rs_fc5(o_z_rs[:, -1:]))
    #     o_z_rs = torch.cat((o, z, rs), dim=-1)
    #
    #     quat_pre_normalized = quat[:, :, :(self.V - 1) * self.D].contiguous()
    #     quat = quat_pre_normalized.view(-1, self.D)
    #     quat = F.normalize(quat, dim=1).view(quat_pre_normalized.shape)
    #
    #     if return_prenorm:
    #         return quat, o_z_rs, quat_h, quat_pre_normalized
    #     else:
    #         return quat, o_z_rs, quat_h

    def forward(self, text, intended_emotion=None, intended_polarity=None,
                acting_task=None, gender=None, age=None, handedness=None, native_tongue=None,
                quat=None, offset_lengths=None, only_encoder=False, only_decoder=False):
        if not only_decoder:
            if self.text_mask is None or self.text_mask.size(0) != text.shape[1]:
                self.text_mask = self._generate_square_subsequent_mask(text.shape[1]).to(text.device)

            text_embed = self.text_embedding(text) * math.sqrt(self.text_dim)
            text_pos_enc = self.text_pos_encoder(text_embed)
            text_latent = self.transformer_encoder(text_pos_enc.permute(1, 0, 2).float(), self.text_mask)
            time_steps = text_latent.shape[0]
            text_latent = self.text_embed(torch.cat((
                text_latent,
                intended_emotion.unsqueeze(0).repeat(time_steps, 1, 1),
                intended_polarity.unsqueeze(0).repeat(time_steps, 1, 1),
                acting_task.unsqueeze(0).repeat(time_steps, 1, 1),
                gender.unsqueeze(0).repeat(time_steps, 1, 1),
                age.unsqueeze(0).repeat(time_steps, 1, 1),
                handedness.unsqueeze(0).repeat(time_steps, 1, 1),
                native_tongue.unsqueeze(0).repeat(time_steps, 1, 1)), dim=-1))
            if only_encoder:
                return text_latent
        else:
            text_latent = text

        offset_lengths = offset_lengths.unsqueeze(0).repeat(text_latent.shape[0], 1, 1)
        gestures_latent = self.text_offsets_to_gestures(torch.cat((text_latent, offset_lengths), dim=-1))
        if self.quat_mask is None or self.quat_mask.size(0) != quat.shape[1]:
            self.quat_mask = self._generate_square_subsequent_mask(quat.shape[1]).to(quat.device)

        quat_pos_enc = self.quat_pos_encoder(quat)
        quat_pred_pre_norm = self.transformer_decoder(quat_pos_enc.permute(1, 0, 2),
                                                      gestures_latent, tgt_mask=self.quat_mask).permute(1, 0, 2)
        if quat_pred_pre_norm.shape[1] == self.T:
            for smoothing_layer in self.temporal_smoothing:
                quat_pred_pre_norm = smoothing_layer(quat_pred_pre_norm)
        quat_pred = quat_pred_pre_norm.contiguous().view(-1, self.quat_channels)
        quat_pred = F.normalize(quat_pred, dim=1).view(quat_pred_pre_norm.shape)
        return quat_pred, quat_pred_pre_norm
