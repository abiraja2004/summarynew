__author__ = 'HyNguyen'

import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import sgd, adadelta
from sklearn import preprocessing
import time
import sys
from nnlayers import RegresstionLayer, LenetConvPoolLayer, FullConectedLayer, ProjectionLayer
import pickle
from summarytrain import load_data

if __name__ == "__main__":

    rng = np.random.RandomState(4488)

    X_train, Y_train , X_valid, Y_valid = load_data(rng)

    Y_train_rouge1 = Y_train[:,0]
    Y_train_rouge2 = Y_train[:,1]
    Y_train_rougesu4 = Y_train[:,2]

    Y_valid_rouge1 = Y_valid[:,0]
    Y_valid_rouge2 = Y_valid[:,1]
    Y_valid_rougesu4 = Y_valid[:,2]

    embed_matrix = np.load("data/word2vec/embed_matrix.npy")
    embed_matrix = np.array(embed_matrix,dtype=np.float64)
    print(embed_matrix.shape, embed_matrix.dtype)

    X = T.imatrix("X")
    Y = T.dvector("Y")

    minibatch_size = 100
    sentence_length = X_train.shape[1]
    embsize = embed_matrix.shape[1]
    vocab_size = embed_matrix.shape[0]
    sentence_shape = (minibatch_size, 1, sentence_length, embsize)
    filter_shape = (20, 1, 5, embed_matrix.shape[1])
    pool_size = (3,1)

    params = [None]*6
    with open("model/w2v_rouge2.bin", mode="rb") as f:
        params = pickle.load(f)
    params = [None]*6
    # maybe wrong in 0_index: is UNK work, Yoon Kim set to vector 0
    project_layer = ProjectionLayer(rng,X,vocab_size,embsize,(minibatch_size,sentence_length),embed_matrix=embed_matrix)

    #conv_layer
    conv_layer = LenetConvPoolLayer(rng, project_layer.output,sentence_shape,filter_shape,pool_size,params=params[0:2],activation=T.tanh)

    # hidden_layer
    hidden_input = conv_layer.output.flatten(2)
    hidden_input_shape = (conv_layer.output_shape[0], conv_layer.output_shape[1]*conv_layer.output_shape[2]*conv_layer.output_shape[3])
    hidden_layer = FullConectedLayer(rng,hidden_input,hidden_input_shape[1],100,params=params[2:4],activation=T.tanh)

    # regression_layer
    regession_layer = RegresstionLayer(rng,hidden_layer.output,100,1, params=params[4:6],activation=T.tanh)

    mse = regession_layer.mse(Y)

    cost = mse + 0.001 * (conv_layer.L2 + hidden_layer.L2 + regession_layer.L2)

    valid_model = theano.function([X,Y],[mse,cost])

    predict_function = theano.function([X], regession_layer.y_pred)

    showfunction = theano.function([X], [project_layer.output, conv_layer.conv_out, conv_layer.pooled_out , conv_layer.output, hidden_layer.output, regession_layer.output])

    hy = X_valid[:2]
    proout, conv_out, pool_out, conv_layer_out , hidden_out, regress_out = showfunction(hy)
    print("ttdtilu")


