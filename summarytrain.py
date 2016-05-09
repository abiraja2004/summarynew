__author__ = 'HyNguyen'
import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import sgd, adadelta
from sklearn import preprocessing
import time
import sys
import pickle

def load_data(rng):
    """
    Load data
    Scale Rouge -> [0:1]
    Devide data to X_train, Y_train, X_valid, Y_valid

    Params:
        rng: random stream which fix seed
    Return:
        (X_train, Y_train , X_valid, Y_valid)

    """
    index_duc04 = np.load("data/word2vec/index.duc04.npy")
    score_duc04 = np.load("data/word2vec/score.duc04.npy")

    index_duc05 = np.load("data/word2vec/index.duc05.npy")
    score_duc05 = np.load("data/word2vec/score.duc05.npy")

    index_dailymail = np.load("data/word2vec/index.dailymail.npy")
    score_dailymail = np.load("data/word2vec/score.dailymail.npy")

    X = np.concatenate((index_dailymail,index_duc04,index_duc05))
    Y = np.concatenate((score_dailymail,score_duc04,score_duc05))
    Y = np.array(Y, dtype=np.float32)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

    Y = min_max_scaler.fit_transform(Y)

    with open("model/scaler.pickle",mode="wb") as f:
        pickle.dump(min_max_scaler,f)

    idxs = list(range(0, X.shape[0]))
    rng.shuffle(idxs)

    validnum = int(X.shape[0]*0.8)

    X_train = X[idxs[:validnum]]
    Y_train = Y[idxs[:validnum]]
    X_valid = X[idxs[validnum:]]
    Y_valid = Y[idxs[validnum:]]

    sys.stdout.write("{0}, {1}, {2}, {3}".format(X_train.shape, X_train.dtype , Y_train.shape, Y_train.dtype))
    sys.stdout.write("{0}, {1}, {2}, {3}".format(X_valid.shape, X_valid.dtype, Y_valid.shape, Y_valid.dtype))
    sys.stdout.flush()
    return X_train, Y_train , X_valid, Y_valid

from nnlayers import LeNetConvPoolLayer, RegressionNeuralNetwork, ProjectionLayer

if __name__ == "__main__":

    rng = np.random.RandomState(4488)
    #
    # X_train, Y_train , X_valid, Y_valid = load_data(rng)
    # Y_train_rouge1 = Y_train[:,0]
    # Y_train_rouge2 = Y_train[:,1]
    # Y_train_rougesu4 = Y_train[:,2]
    #
    # Y_valid_rouge1 = Y_valid[:,0]
    # Y_valid_rouge2 = Y_valid[:,1]
    # Y_valid_rougesu4 = Y_valid[:,2]
    #
    embed_matrix = np.load("data/word2vec/embed_matrix.npy")
    embed_matrix = np.array(embed_matrix,dtype=np.float32)
    sys.stdout.write("vocab_size {0}, embsize {1}\n".format(embed_matrix.shape, embed_matrix.dtype))
    sys.stdout.flush()
    #
    # X = T.imatrix("X")
    # Y = T.fvector("Y")
    #
    # minibatch_size = 100
    # sentence_length = X_train.shape[1]
    # embsize = embed_matrix.shape[1]
    # vocab_size = embed_matrix.shape[0]

    embsize = 300
    img_w = embsize
    img_h = 80
    batch_size = 50
    filter_w = img_w
    filter_shapes = []
    pool_sizes = []
    feature_maps = 100
    filter_hs = [3,4,5]
    conv_non_linear = "relu"
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))

    print(filter_shapes)
    print(pool_sizes)

    vocab_size = 69379

    X = T.imatrix("X")
    Y = T.dvector("Y")

    projection_layer = ProjectionLayer(rng=rng, input=X, vocab_size=vocab_size, embsize=embsize, input_shape=(batch_size,img_h), embed_matrix=embed_matrix)
    print(projection_layer.output_shape)

    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=projection_layer.output,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)

    layer1_input = T.concatenate(layer1_inputs,1)
    input_dims = feature_maps*len(filter_hs)





