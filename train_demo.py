import models
from fewshot_re_kit.data_loader import JSONFileDataLoader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder 
from models.proto import Proto
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
from models.siamese import Siamese
import sys
from torch import optim

model_name = 'bert+siamese'
N = 5
K = 5
if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    K = int(sys.argv[3])

print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))

max_length = 40
train_data_loader = JSONFileDataLoader('./my_data/82/train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./my_data/82/valid.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader = JSONFileDataLoader('./my_data/82/test.json', './data/glove.6B.50d.json', max_length=max_length)

framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
sentence_encoder = CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)

if model_name == 'proto':
    model = Proto(sentence_encoder)
    framework.train(model, model_name, 32, 8, N, K, 1, learning_rate=1e-2, weight_decay=0, optimizer=optim.Adam, train_iter=15000)
elif model_name == 'gnn':
    model = GNN(sentence_encoder, N)
    framework.train(model, model_name, 2, N, N, K, 1, learning_rate=1e-3, weight_decay=0, optimizer=optim.Adam, train_iter=10000)
elif model_name == 'snail':
    print("HINT: SNAIL works only in PyTorch 0.3.1")
    model = SNAIL(sentence_encoder, N, K)
    framework.train(model, model_name, 25, N, N, K, 1, learning_rate=1e-2, weight_decay=0, optimizer=optim.SGD, train_iter=10000)
elif model_name == 'metanet':
    model = MetaNet(N, K, train_data_loader.word_vec_mat, max_length)
    framework.train(model, model_name, 1, N, N, K, 1, learning_rate=5e-3, weight_decay=0, optimizer=optim.Adam)
elif model_name == 'siamese':
    model = Siamese(sentence_encoder)
    framework.train(model, model_name, 1, N, N, K, 5, learning_rate=1e-3, weight_decay=0, optimizer=optim.Adam, train_iter=10000)
elif model_name == 'bert+siamese':
    pretrain_ckpt = 'bert-base-uncased'
    print('加载成功')
    entence_encoder = BERTSentenceEncoder(pretrain_ckpt, max_length)
    model = Siamese(sentence_encoder)
    framework.train(model, model_name, 1, N, N, K, 1, learning_rate=1e-3, weight_decay=1e-3, optimizer=optim.AdamW, train_iter=30000)
else:
    raise NotImplementedError

