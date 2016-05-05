__author__ = 'HyNguyen'

import numpy as np
from theano.tensor.nnet import conv
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from lasagne.updates import sgd, adadelta
from sklearn import preprocessing
import pickle

class LenetConvPoolLayer(object):

    def __init__(self, rng, input, image_shape, filter_shape, poolsize , border_mode ='valid' , activation = T.nnet.sigmoid, params = [None, None]):
        self.input = input
        self.image_shape = image_shape
        # image_shape = (mini_batch_size,1,sentence_length, embed_size)
        # filter_shape: (20,1,5,100)
        self.filter_shape = filter_shape
        self.poolsize = poolsize
        self.activation = activation
        self.output_shape = (image_shape[0],filter_shape[0],int((image_shape[2]-filter_shape[2]+1)/poolsize[0]),int(image_shape[3]-filter_shape[3]+1)/poolsize[1])

        assert image_shape[1] == filter_shape[1]
        self.input = input

        if params[0] is None:
            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
            # initialize weights with random weights
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                              dtype=np.float64),
                                   borrow=True)

            b_values = np.zeros((self.output_shape[1],self.output_shape[2],self.output_shape[3]), dtype=np.float64)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.W, self.b = params[0], params[1]

         # convolve input feature maps with filters
        self.conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape, border_mode=border_mode)

        # downsample each feature map individually, using maxpooling
        self.pooled_out = pool.pool_2d(input=self.conv_out, ds=poolsize, ignore_border=True)

        self.output = self.activation(self.pooled_out + self.b)

        #todo phat sinh lai b bang np.rand

        self.params = [self.W, self.b]

        self.L2 = (self.W**2).sum()

class ProjectionLayer(object):
    def __init__(self,rng ,input, vocab_size, embsize, input_shape, params = [None], embed_matrix = None):
        if embed_matrix is None:
            if params[0] is None:

                self.words_embedding = theano.shared(value= np.asarray(rng.normal(0,0.1,(vocab_size,embsize)),dtype=np.float64),
                                       name = "wordembedding",
                                       borrow=True)
            else:
                self.words_embedding = params[0]
        else:
            self.words_embedding = theano.shared(value= np.asarray(embed_matrix,dtype=np.float64),
                                   name = "embed_matrix",
                                   borrow=True)

        self.output_shape = (input_shape[0],1,input_shape[1],embsize)
        self.output = self.words_embedding[T.cast(input.flatten(1),dtype="int32")].reshape(self.output_shape)
        self.params = [self.words_embedding]
        self.L2 = (self.words_embedding**2).sum()

class FullConectedLayer(object):
    def __init__(self, rng ,input, n_in, n_out, activation = T.nnet.sigmoid, params = [None,None]):
        if params[0] == None:
            self.W = theano.shared(value= np.asarray(rng.rand(n_in,n_out)/np.sqrt(n_in+1),dtype=np.float64),
                                   name = "W",
                                   borrow=True)
            self.b = theano.shared(value= np.asarray(rng.rand(n_out,) ,dtype=np.float64),
                                   name ="b",
                                   borrow=True
            )
        else:
            self.W, self.b = params[0], params[1]

        self.input = input
        self.output = activation(T.dot(input,self.W) + self.b)
        self.params = [self.W, self.b]
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W**2).sum()


class RegresstionLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation = T.nnet.sigmoid, params = [None, None]):
        if params[0] == None:
            self.W = theano.shared(value= np.asarray(rng.rand(n_in,n_out)/np.sqrt(n_in+1),dtype=np.float64),
                                   name = "W",
                                   borrow=True)
            self.b = theano.shared(value= np.asarray(rng.rand(n_out,) ,dtype=np.float64),
                                   name ="b",
                                   borrow=True
            )
        else:
            self.W, self.b = params[0], params[1]

        self.input = input
        self.output = activation(T.dot(input,self.W) + self.b)
        self.y_pred = self.output.flatten(1)
        self.params = [self.W, self.b]
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W**2).sum()

    def mse(self, y):
        return T.mean((self.y_pred - y) ** 2)

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
    Y = np.array(Y,dtype=np.float64)
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

    print(X_train.shape, X_train.dtype , Y_train.shape, Y_train.dtype)
    print(X_valid.shape, X_valid.dtype, Y_valid.shape, Y_valid.dtype)

    return X_train, Y_train , X_valid, Y_valid

import time
import sys

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
    vocab_size = embed_matrix[0]
    sentence_shape = (minibatch_size, 1, sentence_length, embsize)
    filter_shape = (20, 1, 5, embed_matrix.shape[1])
    pool_size = (2,1)

    # maybe wrong in 0_index: is UNK work, Yoon Kim set to vector 0
    project_layer = ProjectionLayer(rng,X,vocab_size,embsize,(minibatch_size,sentence_length),embed_matrix=embed_matrix)

    #conv_layer
    conv_layer = LenetConvPoolLayer(rng, project_layer.output,sentence_shape,filter_shape,pool_size)

    # hidden_layer
    hidden_input = conv_layer.output.flatten(2)
    hidden_input_shape = (conv_layer.output_shape[0], conv_layer.output_shape[1]*conv_layer.output_shape[2]*conv_layer.output_shape[3])
    hidden_layer = FullConectedLayer(rng,hidden_input,hidden_input_shape[1],100)

    # regression_layer
    regession_layer = RegresstionLayer(rng,hidden_layer.output,100,1)

    mse = regession_layer.mse(Y)

    cost = mse + 0.001 * (conv_layer.L2 + hidden_layer.L2 + regession_layer.L2)

    params = conv_layer.params + hidden_layer.params + regession_layer.params

    updates = sgd(cost,params,0.01)

    train_model = theano.function([X,Y],[mse,cost],updates=updates)
    valid_model = theano.function([X,Y],[mse,cost])

    showfunction = theano.function([X,Y],[hidden_input, hidden_layer.output, regession_layer.y_pred, mse, cost])

    # a,b,c,d,e = showfunction(X_train[:100],Y_train_rouge1[:100])
    #
    # print(a,b,c,d,e)

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
            mnb_Y = Y_train_rouge2[train_rand_idxs[batch_i*minibatch_size: batch_i*minibatch_size + minibatch_size]]
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
            mnb_Y  = Y_train_rouge2[valid_rand_idxs[batch_i*minibatch_size: batch_i*minibatch_size + minibatch_size]]
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
            with open("model/w2v_rouge2.bin", mode="wb") as f:
                pickle.dump(params,f)
        else:
            sys.stdout.write("\n")
            patience +=1
        sys.stdout.write("\n")






