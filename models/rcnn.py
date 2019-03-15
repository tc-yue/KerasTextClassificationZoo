# -*- coding: utf-8 -*-
# @Time    : 2018/9/23 10:22
# @Author  : Tianchiyue
# @File    : rcnn.py
# @Software: PyCharm Community Edition

from models.model import BaseModel
from keras.layers import Conv1D, GlobalMaxPooling1D, concatenate, Bidirectional, GRU, LSTM, GlobalAveragePooling1D,\
    SpatialDropout1D, Input,Embedding, CuDNNGRU, CuDNNLSTM


class RCnn(BaseModel):
    def build(self, embedding_matrix):
        self.sentence_input = Input(shape=(self.config['max_length'],),
                                    dtype='int32',
                                    name='sentence_input')
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
        embed = Embedding(embedding_matrix.shape[0],
                          embedding_matrix.shape[1],
                          trainable=self.config['embed_trainable'],
                          weights=[embedding_matrix]
                          )(self.sentence_input)
        embed = SpatialDropout1D(self.config['spatial_dropout_rate'])(embed)
        if self.config['bidirectional']:
            rnn_out = Bidirectional(RNN)(embed)
        else:
            rnn_out = RNN(embed)

        conv = Conv1D(self.config['filters'],
                      kernel_size=self.config['kernel_size'],
                      activation='relu')(rnn_out)
        max_pooling = GlobalMaxPooling1D()(conv)
        avg_pooling = GlobalAveragePooling1D()(conv)
        merged = concatenate([max_pooling, avg_pooling], axis=1)
        return merged