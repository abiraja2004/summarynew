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
    Y = np.array(Y, dtype=np.float64)
    min_max_scaler = preprocessing.MinMaxScaler()

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

    sys.stdout.write("{0}, {1}, {2}, {3}\n".format(X_train.shape, X_train.dtype , Y_train.shape, Y_train.dtype))
    sys.stdout.write("{0}, {1}, {2}, {3}\n".format(X_valid.shape, X_valid.dtype, Y_valid.shape, Y_valid.dtype))
    sys.stdout.flush()
    return X_train, Y_train , X_valid, Y_valid

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

    mse = regess.entropy(Y)

    L2 = sum([conv_layer.L2 for conv_layer in conv_layers]) + regess.L2

    cost = mse + L2

    params = regess.params
    for conv_layer in conv_layers:
        params+=conv_layer.params

    updates = adadelta(cost,params)

    train_model = theano.function([X,Y],[mse, cost],updates=updates)
    valid_model = theano.function([X,Y],[mse, cost])

    showfunction = theano.function(inputs=[X],outputs=regess.regressionlayer.y_pred)

    patience = 0
    best_valid_mse_global = 100
    early_stop = 20
    epoch_i = 0

    train_rand_idxs = list(range(0,X_train.shape[0]))
    valid_rand_idxs = list(range(0,X_valid.shape[0]))

    while patience < early_stop:
        epoch_i +=1

        train_mses = []
        train_costs = []

        valid_mses = []
        valid_costs = []

        best_train_mse = 100
        rng.shuffle(train_rand_idxs)
        batch_number = int(X_train.shape[0] / batch_size)
        start_train = time.clock()
        for batch_i in range(batch_number):
            mnb_X = X_train[train_rand_idxs[batch_i*batch_size: batch_i*batch_size + batch_size]]
            mnb_Y = Y_train_rouge2[train_rand_idxs[batch_i*batch_size: batch_i*batch_size + batch_size]]
            train_mse, train_cost = train_model(mnb_X, mnb_Y)
            train_mse = float(train_mse)
            if train_mse < best_train_mse:
                best_train_mse = train_mse
            train_mses.append(float(train_mse))
            train_costs.append(float(train_cost))
            sys.stdout.write("\rTraining:\t\t batch {0}/{1}:\t\t TrainMSE {2},\t\t BestMSE {3}".format(batch_i+1,batch_number,round(float(train_mse),6),round(best_train_mse,6)))
        end_train = time.clock()

        sys.stdout.write("\n")

        best_valid_mse = 100
        rng.shuffle(valid_rand_idxs)
        batch_number = int(X_valid.shape[0] / batch_size)
        for batch_i in range(batch_number):
            mnb_X = X_valid[valid_rand_idxs[batch_i*batch_size: batch_i*batch_size + batch_size]]
            mnb_Y  = Y_train_rouge2[valid_rand_idxs[batch_i*batch_size: batch_i*batch_size + batch_size]]
            valid_mse, valid_cost = valid_model(mnb_X, mnb_Y)
            valid_mse = float(valid_mse)
            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
            valid_mses.append(float(valid_mse))
            valid_costs.append(float(valid_cost))
            sys.stdout.write("\rValidation:\t\t batch {0}/{1}:\t\t ValidMSE {2},\t\t BestMSE {3}".format(batch_i+1,batch_number,round(float(valid_mse),6),round(best_valid_mse,6)))

        sys.stdout.write("\n")

        aver_train_mse = round(np.mean(np.array(train_mses)),6)
        aver_train_cost = round(np.mean(np.array(train_costs)),6)
        aver_val_mse = round(np.mean(np.array(valid_mses)),6)
        aver_val_cost = round(np.mean(np.array(valid_costs)),6)


        sys.stdout.write("Epoch {0},\t TrainMSE {1},\t TrainCOST {2},\t ValidationMSE {3},\t ValidationCOST {4},\t Patience {5}\n".format(epoch_i,aver_train_mse,aver_train_cost,aver_val_mse,aver_val_cost, patience))

        sys.stdout.flush()


        if aver_val_mse < best_valid_mse_global:
            best_valid_mse_global = aver_val_mse
            sys.stdout.write(" __best__ \n")
            patience = 0
            with open("model/w2v_rouge2.bin", mode="wb") as f:
                pickle.dump(params,f)
        else:
            sys.stdout.write("\n")
            patience +=1
        sys.stdout.write("\n")

        pred = showfunction(X_valid[:10])
        print(pred)
        print(Y_valid_rouge2[:10])
