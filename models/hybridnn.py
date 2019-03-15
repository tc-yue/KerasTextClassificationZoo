# -*- coding: utf-8 -*-
# @Time    : 2018/9/23 10:26
# @Author  : Tianchiyue
# @File    : hybridnn.py
# @Software: PyCharm Community Edition

from models.model import BaseModel
from keras.layers import Bidirectional, GRU, LSTM, SpatialDropout1D, Input,Embedding, CuDNNGRU, CuDNNLSTM, Flatten, \
    Conv1D, concatenate, TimeDistributed, Dense, Activation, Lambda
import keras.backend as K
from models.layers import Capsule


class HybridNN(BaseModel):
    """
    Ref:https://github.com/ShawnyXiao/2018-DC-DataGrand-TextIntelProcess  HybridNN-3
    """

    def build(self, embedding_matrix):
        if self.config['rnn'] == 'gru' and self.config['gpu']:
            RNN = CuDNNGRU(self.config['rnn_output_size'], return_sequences=True)
        elif self.config['rnn'] == 'lstm' and self.config['gpu']:
            RNN = CuDNNLSTM(self.config['rnn_output_size'], return_sequences=True)
        elif self.config['rnn'] == 'gru' and not self.config['gpu']:
            RNN = GRU(self.config['rnn_output_size'], return_sequences=True, dropout=self.config['dropout_rate'],
                      recurrent_dropout=self.config['dropout_rate'])
        else:
            RNN = LSTM(self.config['rnn_output_size'], return_sequences=True, dropout=self.config['dropout_rate'],
                       recurrent_dropout=self.config['dropout_rate'])
        self.sentence_input = Input(shape=(self.config['max_length'],),
                                    dtype='int32',
                                    name='sentence_input')
        embed = Embedding(embedding_matrix.shape[0],
                          embedding_matrix.shape[1],
                          trainable=self.config['embed_trainable'],
                          weights=[embedding_matrix]
                          )(self.sentence_input)
        embed = SpatialDropout1D(self.config['spatial_dropout_rate'])(embed)
        convs = []
        for ksz in self.config['kernel_sizes']:
            conv = Conv1D(self.config['filters'],ksz,activation='relu',padding='same')(embed)
            convs.append(conv)
        cnn_out = concatenate(convs,axis=-1)

        if self.config['bidirectional']:
            rnn_out = Bidirectional(RNN)(embed)
        else:
            rnn_out = RNN(embed)

        capsule_cnn = Capsule(num_capsule=self.config['num_capsule'], dim_capsule=self.config['dim_capsule'],
                              routings=self.config['routings'], share_weights=True, name='capsule_cnn')(cnn_out)
        capsule_cnn = Flatten()(capsule_cnn)

        capsule_rnn = Capsule(num_capsule=self.config['num_capsule'], dim_capsule=self.config['dim_capsule'],
                              routings=self.config['routings'], share_weights=True, name='capsule_rnn')(rnn_out)
        capsule_rnn = Flatten()(capsule_rnn)

        cnn_u = TimeDistributed(Dense(self.config['hidden_dims'], activation='tanh', use_bias=True))(cnn_out)
        cnn_alpha = Dense(1)(cnn_u)
        cnn_alpha = Flatten()(cnn_alpha)
        cnn_alpha = Activation(activation='softmax')(cnn_alpha)
        cnn_att_rep = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 1]))([cnn_out, cnn_alpha])

        rnn_u = TimeDistributed(Dense(self.config['hidden_dims'], activation='tanh', use_bias=True))(rnn_out)
        rnn_alpha = Dense(1)(rnn_u)
        rnn_alpha = Flatten()(rnn_alpha)
        rnn_alpha = Activation(activation='softmax')(rnn_alpha)
        rnn_att_rep = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 1]))([rnn_out, rnn_alpha])

        cnn_concat = concatenate([capsule_cnn, cnn_att_rep], axis=-1)
        rnn_concat = concatenate([capsule_rnn, rnn_att_rep], axis=-1)

        rep = concatenate([cnn_concat,rnn_concat], axis=-1)
        return rep