# -*- coding: utf-8 -*-
# @Time    : 2018/9/23 10:26
# @Author  : Tianchiyue
# @File    : attention.py
# @Software: PyCharm Community Edition


from models.model import BaseModel
from keras.layers import Bidirectional, LSTM, GRU, TimeDistributed, Dense, Flatten, Activation, \
    Lambda, Input, Embedding, dot
import keras.backend as K
from models.layers import LocationAttentionLayer, ClearMaskLayer,BiLocationAttentionLayer, SelfAttentionLayer


class Attention(BaseModel):
    def build(self, embedding_matrix):
        self.sentence_input = Input(shape=(self.config['max_length'],),
                                    dtype='int32',
                                    name='sentence_input')
        embed = Embedding(embedding_matrix.shape[0],
                          embedding_matrix.shape[1],
                          trainable=self.config['embed_trainable'],
                          weights=[embedding_matrix],
                          mask_zero=True
                          )(self.sentence_input)
        if self.config['bidirectional']:
            if self.config['rnn'] == 'gru':
                rnn_out = Bidirectional(GRU(self.config['rnn_output_size'],
                                            return_sequences=True,
                                            dropout=self.config['dropout_rate'],
                                            recurrent_dropout=self.config['dropout_rate']))(embed)
            else:
                rnn_out = Bidirectional(LSTM(self.config['rnn_output_size'],
                                             return_sequences=True,
                                             dropout=self.config['dropout_rate'],
                                             recurrent_dropout=self.config['dropout_rate']))(embed)
        else:
            if self.config['rnn'] == 'gru':
                rnn_out = GRU(self.config['rnn_output_size'],
                              return_sequences=True,
                              dropout=self.config['dropout_rate'],
                              recurrent_dropout=self.config['dropout_rate'])(embed)
            else:
                rnn_out = LSTM(self.config['rnn_output_size'],
                               return_sequences=True,
                               dropout=self.config['dropout_rate'],
                               recurrent_dropout=self.config['dropout_rate'])(embed)

        # word_att = LocationAttentionLayer()(dense_out)
        # sent_rep = dot([dense_out, word_att], axes=[1, 1])
        if self.config['self_att']:
            M = SelfAttentionLayer()(rnn_out)
            rep = Flatten()(M)
        elif self.config['use_mask']:
            alpha = BiLocationAttentionLayer()(rnn_out)
            rep = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 1]))([rnn_out, alpha])
        else:
            u = TimeDistributed(Dense(self.config['hidden_dims'], activation='tanh', use_bias=True))(rnn_out)
            alpha = Dense(1)(u)
            alpha = ClearMaskLayer()(alpha)
            alpha = Flatten()(alpha)
            alpha = Activation(activation='softmax')(alpha)
            rep = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 1]))([rnn_out, alpha])
        return rep