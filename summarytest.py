__author__ = 'HyNguyen'

import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import sgd, adadelta
from sklearn import preprocessing
import time
import sys
import pickle
from summarytrain import load_data
from nnlayers import ProjectionLayer,LeNetConvPoolLayer,RegressionNeuralNetwork,RegresstionLayer,HiddenLayer,Sigmoid,Tanh,ReLU

if __name__ == "__main__":

    rng = np.random.RandomState(4488)

    X_train, Y_train , X_valid, Y_valid = load_data(rng)
    Y_train_rouge1 = Y_train[:,0]
    Y_train_rouge2 = Y_train[:,1]
    Y_train_rougesu4 = Y_train[:,2]

    Y_valid_rouge1 = Y_valid[:,0]
    Y_valid_rouge2 = Y_valid[:,1]
    Y_valid_rougesu4 = Y_valid[:,2]
    #
    embed_matrix = np.load("data/word2vec/embed_matrix.npy")
    embed_matrix = np.array(embed_matrix,dtype=np.float32)
    sys.stdout.write("vocab_size {0}, dtype {1}\n".format(embed_matrix.shape, embed_matrix.dtype))
    sys.stdout.flush()

    xxx = Y_train_rouge1[:50]

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

    X = T.matrix("X")
    Y = T.dvector("Y")

    with open("model/w2v_rouge2.bin", mode="rb") as f:
        save_params = pickle.load(f)

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

    regess = RegressionNeuralNetwork(rng, input=layer1_input,n_in=input_dims,n_hidden=100,n_out=1,activation=[Sigmoid,Sigmoid])

    regess.set_params(save_params[:4])
    for i in xrange(len(filter_hs)):
        print(4+i*2)
        print(4+i*2+2)
        conv_layers[i].set_params(save_params[4+i*2:4+i*2+2])

    mse = regess.mse(Y)

    L2 = sum([conv_layer.L2 for conv_layer in conv_layers]) + regess.L2

    cost = mse + L2

    params = regess.params
    for conv_layer in conv_layers:
        params+=conv_layer.params

    updates = adadelta(cost,params)

    train_model = theano.function([X,Y],[mse, cost],updates=updates)
    valid_model = theano.function([X,Y],[mse, cost])

    showfunction = theano.function(inputs=[X],outputs=regess.hiddenlayer.output)

    X_mnb = X_valid[:batch_size]
    Y_mnb = Y_valid_rouge2[:batch_size]
    print(X_mnb.shape, X_mnb.dtype, Y_mnb, Y_mnb.dtype)
    pred = showfunction(X_mnb)
    print pred
    print Y_valid_rouge2[:batch_size]




