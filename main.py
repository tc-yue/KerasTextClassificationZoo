from models import configs
from models import cnn, rcnn, capsule, hybridnn
from models.attention import Attention
import numpy as np
import pickle
from keras.backend.tensorflow_backend import set_session
import os
import random as rn
import tensorflow as tf
import keras.backend as K
import logging
import sys
import argparse
from collections import Counter
import itertools


def init_env(gpu_id):
    """
    设置gpuid
    :param gpu_id:字符串
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))
    logging.info('GPU%s ready!' % gpu_id)


def rand_set():
    # 设置随机种子
    os.environ['PYTHONHASHSEED'] = '7'
    np.random.seed(7)
    rn.seed(7)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(7)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    logging.info('\t=======Init Over=======')


def load_data(data_path):
    with open(data_path, 'rb')as f:
        data = pickle.load(f)
    logging.info('\t=======Data Loaded=======')
    return data


def load_config(model_name):
    if model_name in ['cnn', 'bicnn']:
        config = configs.cnn_config
    elif model_name in ['deep_cnn']:
        config = configs.dpcnn_config
    elif model_name in ['rnn', 'attention', 'self_attention', 'deep_rnn']:
        config = configs.rnn_config
    elif model_name in ['rcnn']:
        config = configs.rcnn_config
    elif model_name in ['capsule']:
        config = configs.capsule_config
    elif model_name in ['hybrid']:
        config = configs.hybrid_config
    else:
        return None
    logging.info('===={}配置文件加载完毕===='.format(model_name))
    return config


def load_model(model_name, model_config, embedding_matrix):
    if model_name == 'deep_cnn':
        model = cnn.DPCnn(model_config)
    elif model_name == 'cnn':
        model = cnn.Cnn(model_config)
    elif model_name == 'attention':
        model = Attention(model_config)
    elif model_name == 'rcnn':
        model = rcnn.RCnn(model_config)
    elif model_name == 'capsule':
        model = capsule.CapsuleRnn(model_config)
    elif model_name == 'hybrid':
        model = hybridnn.HybridNN(model_config)
    else:
        return None
    model.compile(embedding_matrix)
    logging.info('===={}模型加载完毕===='.format(model_name))
    return model


def train(train_x, train_y, valid_x, valid_y, embedding_matrix, model_list, label_name='reason',
          file_path='trained_models'):
    avg_acc = 0
    for model_name in model_list:
        config = load_config(model_name)
        config['num_classes'] = train_y.shape[1]
        config['max_length'] = train_x.shape[1]
        ytc = load_model(model_name, config, embedding_matrix)
        valid_pred, best_acc = ytc.fit(train_x, train_y, valid_x, valid_y, predicted=True,
                                       filename='{}/{}_{}.model'.format(file_path, label_name, model_name))
        avg_acc += best_acc
        logging.info('\t标签{}\t模型{}得分:\tacc:{}'.format(label_name, model_name, best_acc))
        del ytc
        if K.backend() == 'tensorflow':
            K.clear_session()
            rand_set()
    logging.info('\t标签{}平均分:\tacc:{}'.format(label_name, avg_acc / len(model_list)))


def predict(test_x, embedding_matrix, label_name, model_list, num_classes, use_ensemble=False,
            file_path='trained_models'):
    """
    载入预先训练模型，测试集预测。分为ensemble和single模式
    """
    if not use_ensemble:
        model_name = model_list[0]
        model_filename = '{}/{}_{}.model'.format(file_path, label_name, model_name)
        logging.info('\t开始预测模型{}'.format(model_name))
        config = load_config(model_name)
        config['num_classes'] = num_classes
        config['max_length'] = test_x.shape[1]
        ytc = load_model(model_name, config, embedding_matrix)
        ytc.model.load_weights(model_filename)
        y_pred = ytc.predict(test_x)
        return y_pred
    else:
        all_pred = []
        for model_name in model_list:
            model_filename = '{}/{}_{}.model'.format(file_path, label_name, model_name)
            logging.info('\t开始预测模型{}'.format(model_name))
            config = load_config(model_name)
            config['num_classes'] = num_classes
            config['max_length'] = test_x.shape[1]
            ytc = load_model(model_name, config, embedding_matrix)
            ytc.model.load_weights(model_filename)
            all_pred.append(ytc.predict(test_x))
            del ytc
        return all_pred

def main(argv):
    """
    服务器后台运行: nohup python main.py --mode train --gpu 2  --label slot > out.log 2>&1 &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', help='ensemble,train,test,search_para')
    parser.add_argument('--gpu', default='no', help='before running watch nvidia-smi')
    parser.add_argument('--label', default='reason', help='reason, slot, intent')
    parser.add_argument('--embed', default='domain', help='domain,open,merge')
    parser.add_argument('--cut', default='word', help='char')
    parser.add_argument('--loggingmode', default='info', help='info:only result, debug:training details')
    args = parser.parse_args()
    label_name = args.label
    # cut = args.cut
    emb = args.embed
    all_label = ['reason', 'slot', 'intent']

    if args.loggingmode == 'info':
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s%(message)s',
                            filename='trained_models/{}_{}.log'.format(args.mode, label_name),
                            filemode='a')
    else:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s%(message)s',
                            filename='trained_models/{}_{}.log'.format(args.mode, label_name),
                            filemode='a')
    if len(args.gpu) == 1:
        init_env(str(args.gpu))
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rand_set()

    for cut in ['word', 'char']:
        train_x, train_y, valid_x, valid_y, test_x, domain_emb, open_emb, merge_emb = load_data(
            'data/nn_{}_{}.pkl'.format(label_name, cut))
        model_dict = {
            'intent': ['cnn', 'rcnn', 'attention', 'hybrid'],
        }
        embedding_dict = {
            'domain': domain_emb,
            'open': open_emb,
            'merge': merge_emb
        }

        embedding_matrix = embedding_dict.get(emb)
        model_list = model_dict.get(label_name)
        trained_models_path = 'trained_models/{}_{}_{}'.format(label_name, cut, emb)
        if not os.path.exists(trained_models_path):
            os.mkdir(trained_models_path)

        if args.mode == 'train':
            if label_name == 'all':
                for sub_label in all_label:
                    train(train_x, train_y, valid_x, valid_y, embedding_matrix, model_list,
                          label_name=sub_label, file_path=trained_models_path)
            else:
                train(train_x, train_y, valid_x, valid_y, embedding_matrix, model_list,
                      label_name=label_name, file_path=trained_models_path)

        elif args.mode == 'test':
            predicted_label = predict(test_x, embedding_matrix, label_name, model_list, train_y.shape[1],
                                      use_ensemble=True, file_path=trained_models_path)
            with open('result/test_pred_list_{}_{}.pkl'.format(label_name, cut), 'wb') as f:
                pickle.dump(predicted_label, f)


if __name__ == '__main__':
    main(sys.argv)
