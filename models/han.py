# -*- coding: utf-8 -*-
# @Time    : 2018/7/27 9:24
# @Author  : Tianchiyue
# @File    : han.py
# @Software: PyCharm Community Edition

from models.model import BaseModel
from models.layers import LocationAttentionLayer
from keras.layers import Conv1D, GlobalMaxPooling1D, concatenate, Bidirectional, GRU, LSTM, GlobalAveragePooling1D, \
    SpatialDropout1D, Input,TimeDistributed,Embedding,dot,Dense
from keras.models import Model


class HRnn(BaseModel):
    def build(self,embedding_matrix):
        """
        为了与其他模型一致。 word_input 表示每个句子的输入
        sentence_inpu 表示文档级别输入
        :param embedding_matrix:
        :return:
        """
        word_input = Input(shape=(self.config['max_sent_length'],))
        embed = Embedding(embedding_matrix.shape[0],
                               embedding_matrix.shape[1],
                               trainable=self.config['embed_trainable'],
                               weights=[embedding_matrix]
                               )(word_input)
        embed = SpatialDropout1D(self.config['spatial_dropout'])(embed)
        if self.config['bidirectional']:
            if self.config['rnn'] == 'gru':
                rnn_out = Bidirectional(GRU(self.config['rnn_output_size'],
                                            # return_sequences=True,
                                            dropout=self.config['dropout_rate'],
                                            recurrent_dropout=self.config['dropout_rate']))(embed)
            else:
                rnn_out = Bidirectional(LSTM(self.config['rnn_output_size'],
                                             # return_sequences=True,
                                             dropout=self.config['dropout_rate'],
                                             recurrent_dropout=self.config['dropout_rate']))(embed)
        else:
            if self.config['rnn'] == 'gru':
                rnn_out = GRU(self.config['rnn_output_size'],
                              # return_sequences=True,
                              dropout=self.config['dropout_rate'],
                              recurrent_dropout=self.config['dropout_rate'])(embed)
            else:
                rnn_out = LSTM(self.config['rnn_output_size'],
                               # return_sequences=True,
                               dropout=self.config['dropout_rate'],
                               recurrent_dropout=self.config['dropout_rate'])(embed)
        sent_encoder = Model(word_input, rnn_out)
        self.sentence_input = Input(shape=(self.config['max_sents'], self.config['max_sent_length']))
        doc_encoder = TimeDistributed(sent_encoder)(self.sentence_input)
        doc_rnn_output = Bidirectional(GRU(self.config['rnn_output_size']))(doc_encoder)
        return doc_rnn_output

class Han(BaseModel):
    def build(self,embedding_matrix):
        """
        为了与其他模型一致。 word_input 表示每个句子的输入
        sentence_inpu 表示文档级别输入
        :param embedding_matrix:
        :return:
        """
        word_input = Input(shape=(self.config['max_sent_length'],))
        embed = Embedding(embedding_matrix.shape[0],
                               embedding_matrix.shape[1],
                               trainable=self.config['embed_trainable'],
                               weights=[embedding_matrix]
                               )(word_input)
        embed = SpatialDropout1D(self.config['spatial_dropout'])(embed)
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
        dense_out = TimeDistributed(Dense(self.config['rnn_output_size']))(rnn_out)
        word_att = LocationAttentionLayer()(dense_out)
        sent_rep = dot([dense_out, word_att], axes=[1, 1])
        sent_encoder = Model(word_input, sent_rep)

        self.sentence_input = Input(shape=(self.config['max_sents'], self.config['max_sent_length']))
        doc_encoder = TimeDistributed(sent_encoder)(self.sentence_input)
        doc_rnn_output = Bidirectional(GRU(self.config['rnn_output_size'], return_sequences=True))(doc_encoder)
        doc_dense_out = TimeDistributed(Dense(self.config['rnn_output_size']))(doc_rnn_output)
        sent_att = LocationAttentionLayer()(doc_dense_out)
        sent_rep = dot([doc_dense_out, sent_att], axes=[1, 1])

        return sent_rep