#### 项目介绍
文本分类模型，Keras-TensorFlow

#### 软件架构
```
|-- data
    |-- raw_df.pkl      # intent_train_df,intent_test_df,reason_train_df,reason_test_df
    |-- token2index.pkl # intent_word2index,
|-- models
    |-- attention.py  # attention,self attentive
    |-- capsule.py    # capsule net
    |-- cnn.py        # cnn, DPCNN
    |-- configs.py    # models configuration
    |-- layers.py     # customize layer
    |-- model.py      # base model
    |-- rcnn.py       # rcnn,crnn
|-- utils
    |-- data_process.py  # 处理数据
    |-- dataset.py    # 数据读取
    |-- number.py     # 处理文本中的数字
    |-- score.py      # 官方评估代码
|-- word_vectors
    |-- word_embedding.py  # pretrained word2vec
|-- inference.py      # 少量数据验证
|-- main.py           # 模型训练和测试
|-- postprocessing.py # 生成提交结果
|-- preprocessing.py  # 预处理
```


#### Referenced Paper
1. Joint embedding of words and labels for text classification ACL 2018
2. A structured self-attentive sentence embedding        ICLR 2017
3. Hierarchical attention networks for Document classification  NAACL 2016
4. Deep Pyramid Convolutional Neural Networks for Text Categorization ACL 2017
6. Document Modeling with Gated Recurrent Neural Network for Sentiment Classification ACL2015
7. Recurrent Convolutional Neural Networks for Text Classification  AAAI 2015
10. Character-level Convolutional Networks for Text Classification. NIPS 2015
12. Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms ACL2018

#### 使用说明
1. python preprocessing.py   为NN模型预处理数据
2. python main.py --mode train  训练NN模型
3. python main.py --mode test   测试集预测输出

#### TODO
- [ ] 整理代码

