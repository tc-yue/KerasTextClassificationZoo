# -*- coding: utf-8 -*-
# @Time    : 2018/9/23 10:24
# @Author  : Tianchiyue
# @File    : cnn.py
# @Software: PyCharm Community Edition

from models.model import BaseModel
from keras.layers import Conv1D, GlobalMaxPooling1D, concatenate, BatchNormalization, Activation, PReLU, add, \
    MaxPooling1D, SpatialDropout1D, Dense, Input, Embedding
from keras import regularizers
from models.layers import KMaxPooling


class DPCnn(BaseModel):
    """
    Deep Pyramid cnn
    """

    def block(self, block_input):
        # filters 要与词向量维度相同
        conv1 = Conv1D(self.config['filters'], self.config['kernel_size'], padding='same', activation='linear',
                       kernel_regularizer=self.conv_kern_reg, bias_regularizer=self.bias_kern_reg)(block_input)
        bn1 = BatchNormalization()(conv1)
        pr1 = PReLU()(bn1)
        conv2 = Conv1D(self.config['filters'], self.config['kernel_size'], padding='same', activation='linear',
                       kernel_regularizer=self.conv_kern_reg, bias_regularizer=self.bias_kern_reg)(pr1)
        bn2 = BatchNormalization()(conv2)
        pr2 = PReLU()(bn2)
        added = add([pr2, block_input])
        block_output = MaxPooling1D(pool_size=self.config['max_pool_size'], strides=self.config['max_pool_strides'])(
            added)
        return block_output

    def build(self, embedding_matrix):
        self.sentence_input = Input(shape=(self.config['max_length'],),
                                    dtype='int32',
                                    name='sentence_input')
        embed = Embedding(embedding_matrix.shape[0],
                          embedding_matrix.shape[1],
                          trainable=self.config['embed_trainable'],
                          weights=[embedding_matrix]
                          )(self.sentence_input)
        embed = SpatialDropout1D(self.config['spatial_dropout_rate'])(embed)
        self.conv_kern_reg = regularizers.l2(0.00001)
        self.bias_kern_reg = regularizers.l2(0.00001)
        x = SpatialDropout1D(self.config['spatial_dropout'])(embed)
        for _ in range(self.config['layer_nums']):
            x = self.block(x)
        output = GlobalMaxPooling1D()(x)

        output = Dense(self.config['dense_nr'], activation='linear')(output)
        output = BatchNormalization()(output)
        output = PReLU()(output)
        return output


class Cnn(BaseModel):
    def block(self, block_input, ksz):
        conv = Conv1D(self.config['filters'], ksz)(block_input)
        # 加bn之后学习很慢，而且用relu不收敛
        # bn = BatchNormalization()(conv)
        relu = PReLU()(conv)
        return relu

    def build(self, embedding_matrix):
        self.sentence_input = Input(shape=(self.config['max_length'],),
                                    dtype='int32',
                                    name='sentence_input')
        x = Embedding(embedding_matrix.shape[0],
                      embedding_matrix.shape[1],
                      trainable=self.config['embed_trainable'],
                      weights=[embedding_matrix]
                      )(self.sentence_input)
        x = SpatialDropout1D(self.config['spatial_dropout_rate'])(x)
        convs = []
        for ksz in self.config['kernel_sizes']:
            for _ in range(self.config['layers']):
                x = self.block(x, ksz)
            if self.config['use_kmax_pooling']:
                pooling = KMaxPooling(k=self.config['k'])(x)
            else:
                pooling = GlobalMaxPooling1D()(x)
            convs.append(pooling)
            convs.append(pooling)
        merged = concatenate(convs, axis=-1)
        return merged
