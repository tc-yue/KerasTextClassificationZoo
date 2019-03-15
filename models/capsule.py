# -*- coding: utf-8 -*-
# @Time    : 2018/9/23 10:27
# @Author  : Tianchiyue
# @File    : capsule.py
# @Software: PyCharm Community Edition

from models.model import BaseModel
from keras.layers import Bidirectional, GRU, LSTM, SpatialDropout1D, Input,Embedding, CuDNNGRU, CuDNNLSTM, Flatten
from models.layers import Capsule


class CapsuleRnn(BaseModel):

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
        x = Embedding(embedding_matrix.shape[0],
                      embedding_matrix.shape[1],
                      trainable=self.config['embed_trainable'],
                      weights=[embedding_matrix]
                      )(self.sentence_input)
        embed = SpatialDropout1D(self.config['spatial_dropout_rate'])(x)
        if self.config['bidirectional']:
            rnn_out = Bidirectional(RNN)(embed)
        else:
            rnn_out = RNN(embed)
        capsule = Capsule(num_capsule=self.config['num_capsule'], dim_capsule=self.config['dim_capsule'],
                          routings=self.config['routings'],share_weights=True)(rnn_out)
        rep = Flatten()(capsule)
        return rep
