# -*- coding: utf-8 -*-
# @Time    : 2018/7/27 11:31
# @Author  : Tianchiyue
# @File    : rnn.py
# @Software: PyCharm Community Edition

# -*- coding: utf-8 -*-
# @Time    : 2018/3/28 15:14
# @Author  : Tianchiyue
# @File    : rnn.py
# @Software: PyCharm Community Edition
from models.model import BaseModel
from keras.layers import Bidirectional, LSTM, GRU, GaussianNoise, Dropout, dot, Input, Embedding
from models.layers import LocationAttentionLayer


class Rnn(BaseModel):
    def build(self, embedding_matrix):
        self.sentence_input = Input(shape=(self.config['max_length'],),
                                    dtype='int32',
                                    name='sentence_input')
        embed = Embedding(embedding_matrix.shape[0],
                          embedding_matrix.shape[1],
                          trainable=self.config['embed_trainable'],
                          weights=[embedding_matrix]
                          )(self.sentence_input)
        if self.config['bidirectional']:
            if self.config['rnn'] == 'gru':
                rnn_out = Bidirectional(GRU(self.config['rnn_output_size'],
                                            dropout=self.config['dropout_rate'],
                                            recurrent_dropout=self.config['dropout_rate']))(embed)
            else:
                rnn_out = Bidirectional(LSTM(self.config['rnn_output_size'],
                                             dropout=self.config['dropout_rate'],
                                             recurrent_dropout=self.config['dropout_rate']))(embed)
        else:
            if self.config['rnn'] == 'gru':
                rnn_out = GRU(self.config['rnn_output_size'],
                              dropout=self.config['dropout_rate'],
                              recurrent_dropout=self.config['dropout_rate'])(embed)
            else:
                rnn_out = LSTM(self.config['rnn_output_size'],
                               dropout=self.config['dropout_rate'],
                               recurrent_dropout=self.config['dropout_rate'])(embed)
        return rnn_out


class DeepRnn(BaseModel):
    """
    Ref:https://github.com/cbaziotis/datastories-semeval2017-task4
    Bilayer rnn with attention
    """
    def build(self, embedding_matrix):
        self.sentence_input = Input(shape=(self.config['max_length'],),
                                    dtype='int32',
                                    name='sentence_input')
        embed = Embedding(embedding_matrix.shape[0],
                          embedding_matrix.shape[1],
                          trainable=self.config['embed_trainable'],
                          weights=[embedding_matrix]
                          )(self.sentence_input)
        def rnn_block(sentence):
            lstm_out = Bidirectional(LSTM(self.config['rnn_output_size'],
                                          dropout=self.config['dropout_rate'],
                                          recurrent_dropout=self.config['dropout_rate'],
                                          return_sequences=True))(sentence)
            return lstm_out
        x = GaussianNoise(self.config['noise'])(embed)
        if self.config['dropout_words'] > 0:
            x = Dropout(self.config['dropout_words'])(x)
        for _ in range(self.config['layers']):
            x = rnn_block(x)
        alpha = LocationAttentionLayer()(x)
        rep = dot([x, alpha], axes=[1, 1])
        return rep