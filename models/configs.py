# -*- coding: utf-8 -*-
# @Time    : 2018/7/27 8:42
# @Author  : Tianchiyue
# @File    : configs.py
# @Software: PyCharm Community Edition

hybrid_config = {
    'routings':5,
    'num_capsule':20,
    'dim_capsule':16,
    'spatial_dropout_rate': 0.2,
    'gpu': True,
    'self_att': True,
    'noise': 0.3,
    'clipnorm': 0.1,
    'kernel_sizes':[3,5,7],
    'filters': 256,
    'loss_l2': 0.0001,
    'layers': 2,
    'max_length': 100,
    'use_mask': True,
    'dropout_words': 0.3,
    'dropout_rate': 0.2,
    'num_classes': 2,
    'rnn_output_size': 256,
    'embed_trainable': True,
    'batch_size': 32,
    'optimizer': 'adam',
    'learning_rate': 0.0005,
    'epochs': 10,
    'n_stop': 5,
    'hidden_dims': 256,
    'bidirectional': True,
    'rnn': 'gru',
    'activation': 'tanh',
    'use_mlp': True,
    'use_l2': False,
    'l2': 0.0001,
    'lr_decay_epoch': 3,
    'lr_decay_rate': 0.5,
    'han': True
}

capsule_config = {
    'routings':5,
    'num_capsule':10,
    'dim_capsule':16,
    'spatial_dropout_rate': 0.2,
    'gpu': True,
    'self_att': True,
    'noise': 0.3,
    'clipnorm': 0.1,
    'loss_l2': 0.0001,
    'layers': 2,
    'max_length': 100,
    'use_mask': True,
    'dropout_words': 0.3,
    'dropout_rate': 0.2,
    'num_classes': 2,
    'rnn_output_size': 300,
    'embed_trainable': True,
    'batch_size': 32,
    'optimizer': 'adam',
    'learning_rate': 0.0005,
    'epochs': 15,
    'n_stop': 5,
    'hidden_dims': 300,
    'bidirectional': True,
    'rnn': 'gru',
    'activation': 'tanh',
    'use_mlp': True,
    'use_l2': False,
    'l2': 0.0001,
    'lr_decay_epoch': 3,
    'lr_decay_rate': 0.5,
    'han': True
}

cnn_config = {
    'spatial_dropout_rate': 0.2,
    'gpu': True,
    'filters': 300,
    'kernel_sizes': [2, 3],  # 调整 2
    'layers': 2,  # 调整 2
    'spatial_dropout': 0.2,
    'dense_nr': 300,
    'max_length': 1000,
    'num_classes': 2,
    'dropout_rate': 0.2,
    'embed_trainable': True,
    'batch_size': 32,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'epochs': 15,
    'n_stop': 5,
    'hidden_dims': 300,
    'activation': 'tanh',
    'use_mlp': True,
    'use_l2': False,
    'l2': 0.0001,
    'lr_decay_epoch': 3,
    'lr_decay_rate': 0.5,
    'use_kmax_pooling': False,
    'k': 2
}

dpcnn_config = {
    'spatial_dropout_rate': 0.2,
    'gpu': True,
    'filters': 100,
    'kernel_size': 3,  # 调整 2
    'max_pool_size': 3,  # 调整 2
    'max_pool_strides': 2,
    'spatial_dropout': 0.2,
    'dense_nr': 300,
    'max_length': 1000,
    'layer_nums': 7,
    'num_classes': 2,
    'dropout_rate': 0.2,
    'embed_trainable': True,
    'batch_size': 8,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'epochs': 15,
    'n_stop': 5,
    'hidden_dims': 250,
    'activation': 'tanh',
    'use_mlp': False,
    'use_l2': False,
    'l2': 0.0001,
    'lr_decay_epoch': 3,
    'lr_decay_rate': 0.5,
}

rcnn_config = {
    'spatial_dropout_rate': 0.2,
    'gpu': True,
    'num_classes': 2,
    'dropout_rate': 0.2,
    'rnn_output_size': 300,
    'max_length': 100,
    'embed_trainable': True,
    'filters': 200,
    'kernel_size': 2,
    'batch_size': 32,
    'optimizer': 'adam',
    'learning_rate': 0.0005,
    'epochs': 15,
    'n_stop': 5,
    'hidden_dims': 300,
    'bidirectional': True,
    'rnn': 'lstm',
    'activation': 'tanh',
    'use_mlp': True,
    'use_l2': False,
    'l2': 0.0001,
    'lr_decay_epoch': 3,
    'lr_decay_rate': 0.5,
}



rnn_config = {
    'spatial_dropout_rate': 0.2,
    'gpu': True,
    'self_att': True,
    'noise': 0.3,
    'clipnorm': 0.1,
    'loss_l2': 0.0001,
    'layers': 2,
    'max_length': 100,
    'use_mask': True,
    'dropout_words': 0.3,
    'dropout_rate': 0.2,
    'num_classes': 2,
    'rnn_output_size': 300,
    'embed_trainable': True,
    'batch_size': 32,
    'optimizer': 'adam',
    'learning_rate': 0.0005,
    'epochs': 15,
    'n_stop': 5,
    'hidden_dims': 300,
    'bidirectional': True,
    'rnn': 'lstm',
    'activation': 'tanh',
    'use_mlp': True,
    'use_l2': False,
    'l2': 0.0001,
    'lr_decay_epoch': 3,
    'lr_decay_rate': 0.5,
    'han': True
}



