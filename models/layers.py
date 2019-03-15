# -*- coding: utf-8 -*-
# @Time    : 2018/9/23 10:25
# @Author  : Tianchiyue
# @File    : layers.py
# @Software: PyCharm Community Edition

from keras import backend as K
import tensorflow as tf
from keras import initializers, activations
from keras.engine import Layer, InputSpec
from keras.layers import Flatten, Activation


# capsule net 的挤压激活函数
def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


class Capsule(Layer):
    """
    A Capsule Implement with Pure Keras
    以Num_capsule=10,Dim_capsule=16,routings=5, time_steps=200为例
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)    # bsz,200,160
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))   # bsz,200,10,16
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))   # bsz,10,200,16
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # bsz,10,200
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # bsz,200,10
            c = K.softmax(b)    # bsz,200,10, 对最后一个轴10维进行sofrmax
            c = K.permute_dimensions(c, (0, 2, 1))  # bsz,10,200
            b = K.permute_dimensions(b, (0, 2, 1))  # bsz,200,10
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))  # bsz,10,16
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])   # bsz,10,200

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs, mask=None):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)


class AvgMaskPooling(Layer):
    """
    平均池化加mask
    # Input shape
        batch_size,time_steps,hidden_dims
        batch_size,hidden_dims
    """

    def __init__(self,
                 **kwargs):
        self.support_mask = True
        super(AvgMaskPooling, self).__init__(**kwargs)

    def call(self, x, mask=None):
        float_mask = tf.to_float(mask)
        x_mask = K.expand_dims(float_mask)
        x_sum = x * x_mask
        H_enc = K.sum(x_sum, axis=1, keepdims=True)  # bsz,1,emb
        H_enc = K.squeeze(H_enc, axis=1)

        x_mask_sum = K.sum(x_mask, axis=1, keepdims=True)
        x_mask_sum = K.squeeze(x_mask_sum, axis=1)

        H_enc_1 = H_enc / x_mask_sum
        return H_enc_1

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], input_shape[2]])

    def compute_mask(self, x, mask=None):
        return None


class AttentionLayer(Layer):
    """
    # Input shape
        batch_size,time_steps,hidden_dims
        batch_size,time_steps
    """
    def __init__(self,
                 activation='tanh',
                 use_bias=False,
                 match_func='bilinear',
                 **kwargs):
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.support_mask = True
        self.match_func = match_func
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[0][2], input_shape[0][2]),
                                 initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(1,),
                                        initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                        trainable=True)
        else:
            self.bias = None
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        assert self.match_func in ['bilinear', 'dot']
        h, t = x[0], x[1]
        if self.match_func == 'bilinear':
            hw = K.dot(h, self.W)
            output = K.batch_dot(hw, t, axes=[2, 1])
        if self.match_func == 'dot':
            output = K.batch_dot(h, t, axes=[2, 1])
        # todo match_func concat:vtanh(w[h;t])
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        atten = self.softmask(output, mask[0])
        return atten

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0][0], input_shape[0][1]])

    def compute_mask(self, x, mask=None):
        if mask:
            return mask[0]
        else:
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


class LocationAttentionLayer(Layer):
    """
    # Input shape
        batch_size,time_steps,hidden_dims
    # Output shape
        batch_size,time_steps
    """
    def __init__(self,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.support_mask = True
        super(LocationAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                 shape=(input_shape[2], 1),
                                 initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(1,),
                                        initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                        trainable=True)
        else:
            self.bias = None
        super(LocationAttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.dot(x, self.kernel)
        # todo match_func concat:vtanh(w[h;t])
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        output = K.batch_flatten(output)
        if self.activation is not None:
            output = self.activation(output)
        atten = self.softmask(output, mask)
        return atten

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], input_shape[1]])

    def compute_mask(self, x, mask=None):
        if mask is not None:
            return mask
        else:
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

class BiLocationAttentionLayer(Layer):
    """
    # Input shape
        batch_size,time_steps,hidden_dims
    # Output shape
        batch_size,time_steps
    """
    def __init__(self,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.support_mask = True
        super(BiLocationAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.v = self.add_weight(name='v',
                                 shape=(input_shape[2], 1),
                                 initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 trainable=True)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2], input_shape[2]),
                                      initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(input_shape[2],),
                                        initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                        trainable=True)
        else:
            self.bias = None
        super(BiLocationAttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.dot(x, self.kernel)
        # todo match_func concat:vtanh(w[h;t])
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        output = K.dot(output, self.v)
        output = K.batch_flatten(output)
        atten = self.softmask(output, mask)
        return atten

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], input_shape[1]])

    def compute_mask(self, x, mask=None):
        if mask is not None:
            return mask
        else:
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


# TODO 自定义损失函数
class SelfAttentionLayer(Layer):
    """
    # Input shape
        batch_size,time_steps,hidden_dims
    # Output shape
        batch_size,time_steps
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
        #         self.bsz = input_shape[0]
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
        A = self.softmask(wh2, mask)
        M = K.batch_dot(A, h, axes=[2, 1])  # bsz * r * hidden_dims
        #         self.add_custom_loss(A)
        return M

    #     def add_custom_loss(self,A):
    #         tile_eye = tf.tile(tf.eye(self.r), [self.bsz, 1])
    #         tile_eye = tf.reshape(tile_eye, [-1, self.r, self.r])
    #         A_T = tf.transpose(A, perm=[0, 2, 1])
    #         AA_T = tf.matmul(A, A_T) - tile_eye
    #         P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))
    #         self.add_loss(P)


    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], self.r, input_shape[2]])

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
class ClearMaskLayer(Layer):
    """
    after using a layer that supports masking in keras,
    you can use this layer to remove the mask before softmax layer
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(ClearMaskLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, x, mask=None):
        return None