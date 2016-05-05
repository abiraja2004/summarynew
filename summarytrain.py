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
    embed_matrix = np.array(embed_matrix,dtype=np.float32)
    sys.stdout.write("vocab_size {0}, embsize {1}".format(embed_matrix.shape, embed_matrix.dtype))
    sys.stdout.flush()

    X = T.imatrix("X")
    Y = T.fvector("Y")

    minibatch_size = 100
    sentence_length = X_train.shape[1]
    embsize = embed_matrix.shape[1]
    vocab_size = embed_matrix.shape[0]
    sentence_shape = (minibatch_size, 1, sentence_length, embsize)
    filter_shape = (20, 1, 5, embed_matrix.shape[1])
    pool_size = (3,1)

    # maybe wrong in 0_index: is UNK work, Yoon Kim set to vector 0
    project_layer = ProjectionLayer(rng,X,vocab_size,embsize,(minibatch_size,sentence_length),embed_matrix=embed_matrix)

    #conv_layer
    conv_layer = LenetConvPoolLayer(rng, project_layer.output,sentence_shape,filter_shape,pool_size,activation=T.tanh)

    # hidden_layer
    hidden_input = conv_layer.output.flatten(2)
    hidden_input_shape = (conv_layer.output_shape[0], conv_layer.output_shape[1]*conv_layer.output_shape[2]*conv_layer.output_shape[3])
    hidden_layer = FullConectedLayer(rng,hidden_input,hidden_input_shape[1],100,activation=T.tanh)

    # regression_layer
    regession_layer = RegresstionLayer(rng,hidden_layer.output,100,1,activation=T.tanh)

    mse = regession_layer.mse(Y)

    cost = mse + 0.001 * (conv_layer.L2 + hidden_layer.L2 + regession_layer.L2)

    params = conv_layer.params + hidden_layer.params + regession_layer.params

    updates = adadelta(cost,params)

    train_model = theano.function([X,Y],[mse,cost],updates=updates)
    valid_model = theano.function([X,Y],[mse,cost])

    showfunction = theano.function([X,Y],[hidden_input, hidden_layer.output, regession_layer.y_pred, mse, cost])

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
        batch_number = int(X_train.shape[0] / minibatch_size)
        start_train = time.clock()
        for batch_i in range(batch_number):
            mnb_X = X_train[train_rand_idxs[batch_i*minibatch_size: batch_i*minibatch_size + minibatch_size]]
            mnb_Y = Y_train_rouge1[train_rand_idxs[batch_i*minibatch_size: batch_i*minibatch_size + minibatch_size]]
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
        batch_number = int(X_valid.shape[0] / minibatch_size)
        for batch_i in range(batch_number):
            mnb_X = X_valid[valid_rand_idxs[batch_i*minibatch_size: batch_i*minibatch_size + minibatch_size]]
            mnb_Y  = Y_valid_rouge1[valid_rand_idxs[batch_i*minibatch_size: batch_i*minibatch_size + minibatch_size]]
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

        sys.stdout.write("Epoch {0},\t TrainMSE {1},\t TrainCOST {2},\t ValidationMSE {3},\t ValidationCOST {4},\t Patience {5}".format(epoch_i,aver_train_mse,aver_train_cost,aver_val_mse,aver_val_cost, patience))
        sys.stdout.flush()

        if aver_val_mse < best_valid_mse_global:
            best_valid_mse_global = aver_val_mse
            sys.stdout.write(" __best__ \n")
            patience = 0
            with open("model/w2v_rouge1.bin", mode="wb") as f:
                pickle.dump(params,f)
        else:
            sys.stdout.write("\n")
            patience +=1
        sys.stdout.write("\n")






