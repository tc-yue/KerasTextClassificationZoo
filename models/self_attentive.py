# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 15:51
# @Author  : Tianchiyue
# @File    : self_attentive.py
# @Software: PyCharm Community Edition
from keras.layers import GRU, Bidirectional,Input, Embedding, Flatten,Dense
from keras.models import Model
from keras.engine import Layer
from keras import initializers, activations
import keras.backend as K
import tensorflow as tf


"""
structured self attentive
https://github.com/flrngel/Self-Attentive-tensorflow
"""


class SelfAttentionLayer(Layer):
    """
    # Input shape
        batch_size,time_steps,hidden_dims
    # Output shape
        (batch_size,attention_hops,hidden_dims*2) (batch_size,)
    """

    def __init__(self,
                 attention_unit=100,
                 attention_hops=20,
                 activation="tanh",
                 use_bias=False,
                 **kwargs):
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.da = attention_unit
        self.r = attention_hops
        self.support_mask = True
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ws1 = self.add_weight(name='ws1',
                                   shape=(input_shape[2], self.da),
                                   initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                   trainable=True)
        self.ws2 = self.add_weight(name='ws2',
                                   shape=(self.da, self.r),
                                   initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                   trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(input_shape[2],),
                                        initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                        trainable=True)
        else:
            self.bias = None
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, h, mask=None):
        wh1 = K.dot(h, self.ws1)  # time_steps * da
        if self.activation is not None:
            wh1 = self.activation(wh1)
        wh2 = K.permute_dimensions(K.dot(wh1, self.ws2), (0, 2, 1))  # r * time_steps
        A = self.softmask(wh2, mask)  # bsz * r * time_steps
        M = K.batch_dot(A, h, axes=[2, 1])  # bsz * r * hidden_dims
        #         return [M, K.tile(K.eye(self.r),[K.int_shape(h)[0],1])]
        tile_eye = K.eye(self.r)
        #         tile_eye = K.tile(K.eye(self.r),[K.int_shape(h)[0],1])   # bsz*r,r
        tile_eye = K.reshape(tile_eye, [-1, self.r, self.r])  # bsz,r,r
        AA_ = K.batch_dot(A, A, axes=[2, 2])
        AA_T = AA_ - tile_eye  # bsz *r *r
        P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))  # bsz
        #         self.add_custom_loss(A)
        return [M, K.expand_dims(P)]

    #     def add_custom_loss(self,A):
    #         tile_eye = tf.tile(tf.eye(self.r), [self.bsz, 1])
    #         tile_eye = tf.reshape(tile_eye, [-1, self.r, self.r])
    #         A_T = tf.transpose(A, perm=[0, 2, 1])
    #         AA_T = tf.matmul(A, A_T) - tile_eye
    #         P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))
    #         self.add_loss(P)


    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.r, input_shape[2]), (input_shape[0], 1)]

    def compute_mask(self, x, mask=None):
        #         if mask is not None:
        #             return mask
        #         else:
        #             return None
        return None

    def softmask(self, x, mask, axis=-1):
        """
        softmax with mask, used in attention mechanism others
        :param x:
        :param mask:
        :param axis:
        :return:
        """
        y = K.exp(x)
        if mask is not None:
            mask = K.repeat(mask, self.r)
            y = y * tf.to_float(mask)
        sumx = K.sum(y, axis=axis, keepdims=True) + 1e-6
        x = y / sumx
        return K.relu(x)


x = Input((10,))
embed = Embedding(200,100,mask_zero=True)(x)
bigru = Bidirectional(GRU(100,return_sequences=True))(embed)
M, P = SelfAttentionLayer()(bigru)
M = Flatten()(M)
predict = Dense(10)(M)
model = Model(inputs=[x],outputs=[predict,P])
model.compile(optimizer='adam', loss=['categorical_crossentropy',lambda y_true,y_pred: y_pred], loss_weights=[1.,0.2]
              , metrics=['accuracy']
             )
model.summary()