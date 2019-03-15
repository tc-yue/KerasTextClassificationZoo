# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 16:13
# @Author  : Tianchiyue
# @File    : joint_label.py.py
# @Software: PyCharm Community Edition
from keras.layers import GRU, Bidirectional, Input, Embedding, Lambda, Dense, Conv1D
from keras.models import Model
from keras.engine import Layer
from keras import initializers, activations
import keras.backend as K
import tensorflow as tf

"""
Joint Embedding of Words and Labels for Text Classification
https://github.com/guoyinwang/LEAM
"""


class AttEncode(Layer):
    """
    # Input shape
        batch_size,time_steps,hidden_dims
    # Output shape
        batch_size,time_steps
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.support_mask = True
        super(AttEncode, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttEncode, self).build(input_shape)

    def call(self, x, mask=None):
        sentence_embed = x[0]  # b,s,e
        label_embed_w = x[1]  # c,e

        sentence_embed_norm = K.l2_normalize(sentence_embed, axis=-1)
        label_embed_norm = K.l2_normalize(label_embed_w, axis=-1)

        G = K.dot(sentence_embed, K.transpose(label_embed_norm))  # b,s,c
        conv = Conv1D(self.filters, self.kernel_size, padding="same", activation="relu")(G)
        att_v = K.max(conv, axis=-1)  # b,S
        att_soft = K.softmax(att_v)
        #         att_soft = self.softmask(att_v,mask[0])   #
        H_enc = K.batch_dot(sentence_embed, att_v, axes=[1, 1])
        return H_enc  # b,e

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

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
            y = y * tf.to_float(mask)
        sumx = K.sum(y, axis=axis, keepdims=True) + 1e-6
        x = y / sumx
        return K.relu(x)


class LableEmbedding(Embedding):
    def call(self, inputs):
        return self.embeddings

    def compute_output_shape(self, input_shape):
        return (input_shape[1], self.output_dim)


class DenseLoss(Layer):

    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(DenseLoss, self).__init__(**kwargs)

    def call(self, inputs):
        # 对于center loss来说，返回结果还是跟Dense的返回结果一致
        # 所以还是普通的矩阵乘法加上偏置
        H_enc = inputs[0]
        W_class = inputs[1]
        self.label_true = K.eye(self.num_classes)
        logits = Dense(self.num_classes, activation="softmax")(H_enc)
        self.label_pred = Dense(self.num_classes, activation="softmax")(W_class)
        return logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.num_classes)

    def loss(self, y_true, y_pred, class_penalty=0.5):
        # 定义完整的loss
        y_true = K.cast(y_true, 'int32')  # 保证y_true的dtype为int32
        crossentropy = K.categorical_crossentropy(y_true, y_pred, from_logits=True)
        label_loss = K.mean(K.categorical_crossentropy(self.label_true, self.label_pred))
        return crossentropy + class_penalty + label_loss


sentence_length = 20
num_classes = 10
embedding_dims = 100
sentence_input = Input(shape=(sentence_length,))
sentence_embed = Embedding(1000, embedding_dims,
                           #                        mask_zero=True
                           )(sentence_input)  # b,s,e
label_input = Input(shape=(1,))
weights = LableEmbedding(num_classes, embedding_dims)(label_input)
# label_embed = label_embed_layer(label_input)         #b,e
# weights = Lambda(lambda x:x)(label_embed_layer.weights[0])                #c ,e
H_enc = AttEncode(filters=20, kernel_size=5)([sentence_embed, weights])  # b,e
dloss = DenseLoss(num_classes)
output = dloss([H_enc, weights])
model = Model(inputs=[sentence_input, label_input], outputs=H_enc)
model.compile(loss=dloss.loss, optimizer='adam')

model.summary()
