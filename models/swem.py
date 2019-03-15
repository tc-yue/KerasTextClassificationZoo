# -*- coding: utf-8 -*-
# @Time    : 2018/9/5 18:15
# @Author  : Tianchiyue
# @File    : swem.py.py
# @Software: PyCharm Community Edition

from models.model import BaseModel
from keras.layers import Conv1D, GlobalMaxPooling1D, concatenate, BatchNormalization, Activation, PReLU, add, \
    MaxPooling1D, SpatialDropout1D, Dense, Input, Embedding, AveragePooling1D
from models.layers import ClearMaskLayer, AvgMaskPooling

# TODO 跑的结果不理想，带调试


class SWEM(BaseModel):
    """
    Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms ACL 2018
    Ref: https://github.com/dinghanshen/SWEM
    """
    def build(self, embedding_matrix):
        self.sentence_input = Input(shape=(self.config['max_length'],),
                                    dtype='int32',
                                    name='sentence_input')
        embed = Embedding(embedding_matrix.shape[0],
                          embedding_matrix.shape[1],
                          mask_zero=True,
                          trainable=self.config['embed_trainable'],
                          weights=[embedding_matrix]
                          )(self.sentence_input)
        embed = SpatialDropout1D(rate=self.config['spatial_dropout_rate'])(embed)
        clear_embed = ClearMaskLayer()(embed)
        if self.config['pooling_mode'] == 'concat':
            avg_mask = AvgMaskPooling()(embed)
            max_pool = GlobalMaxPooling1D()(clear_embed)
            rep = concatenate([avg_mask, max_pool])
        elif self.config['pooling_mode'] == 'hier':
            avg_under_pooling = AveragePooling1D(pool_size=3, strides=1, padding='same')(clear_embed)
            rep = GlobalMaxPooling1D()(avg_under_pooling)
        return rep
