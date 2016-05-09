__author__ = 'HyNguyen'

import numpy as np
import activation as act
import utils as u
import time
import pickle


class RecursiveNeuralNetworl(object):
    def __init__(self, act_func = act.sigmoid, embsize = 50, mnb_size = 100, lr = 0.1, l2_reg_level = 0.001, wordvector = None, name ="RecursiveNeuralNetwork"):

        # init wordvector
        self.wordvector = wordvector
        self.embsize = embsize

        # activation function
        self.act_func = act_func

        # output dimension
        self.output_dim = 2 # sentiment 0 1

        # params
        self.mnb_size = mnb_size
        self.lr = lr
        self.l2_reg_level = l2_reg_level
        self.name = name

        self.rng = np.random.RandomState(4488)

        # init weight hidden layer
        self.Wh_l = u.init_w(self.rng,(self.embsize, self.embsize))
        self.Wh_r = u.init_w(self.rng,(self.embsize, self.embsize))
        self.bh   = u.init_b((1,self.embsize))

        # init weight softmax layer
        self.Ws = u.init_w(self.rng,(self.embsize, self.output_dim))
        self.bs = u.init_b((1,self.output_dim))

        # Gradients
        self.dWh_l = np.empty(self.Wh_l.shape)
        self.dWh_r = np.empty(self.Wh_r.shape)
        self.dbh = np.empty(self.bh.shape)
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty(self.bs.shape)

        self.Wh_l_gt = np.empty(self.Wh_l.shape)
        self.Wh_r_gt = np.empty(self.Wh_r.shape)
        self.bh_gt = np.empty(self.bh.shape)
        self.Ws_gt = np.empty(self.Ws.shape)
        self.bs_gt = np.empty(self.bs.shape)

    def set_params(self, params):
        """
        Params
            params: is tuple (self.Wh_l, self.Wh_r, self.bh, self.Ws, self.bs)
        """
        if len(params) == 5:
            self.Wh_l, self.Wh_r, self.bh, self.Ws, self.bs  = params

    def set_params_from_vec(self, vec):
        whl = vec[0:self.embsize**2].reshape(self.embsize,self.embsize)
        whr = vec[self.embsize**2:2*(self.embsize**2)]
        bh  = vec[2*(self.embsize**2):2*(self.embsize**2) + self.embsize]
        Ws  = vec[2*(self.embsize**2)+self.embsize:2*(self.embsize**2)+self.embsize+(self.embsize*self.output_dim)]
        bs  = vec[-self.output_dim:]

    def forward_tree(self, tree):
        if tree.is_leaf():
            tree.p = self.wordvector.wordvector(tree.word).reshape(1,-1)
            return
        self.forward_tree(tree.subtrees[0])
        self.forward_tree(tree.subtrees[1])
        tree.p = self.act_func.activate(np.dot(tree.subtrees[0].p,self.Wh_l)+ np.dot(tree.subtrees[1].p, self.Wh_r) + self.bh)

    def forward_softmax(self, tree):
        Z = np.dot(tree.p, self.Ws) + self.bs
        tree.softmax = act.softmax(Z)

    def forward(self, tree):
        """
        Param
            root: root of NLTK Tree
        Returns
            root_node: InternalNode of binary tree
            softmax_layer: softmax layer
        """
        self.forward_tree(tree)
        self.forward_softmax(tree)
        pred = np.argmax(tree.softmax)
        cost = -np.log(tree.softmax[0,tree.label])
        print(tree.softmax,cost)
        return cost, pred

    def backward_softmax(self, tree):
        # one host label
        t = np.zeros((1,self.output_dim), dtype=np.float32)
        t[:,tree.label] = 1

        # back propagation softmax
        delta = tree.softmax - t
        grad = np.dot(tree.p.T,delta)
        self.dWs += grad
        self.dbs += delta
        # return gradient propagation for previous layer
        back_grad = np.dot(delta, self.Ws.T)
        return back_grad

    def backward_tree(self, tree, back_grad):
        if tree.is_leaf():
            return
        # delta + grand for left side and right side
        delta = back_grad*self.act_func.derivative(tree.p)
        grad_left = np.dot(tree.subtrees[0].p.T,delta)
        grad_right = np.dot(tree.subtrees[1].p.T,delta)
        self.dbh += delta
        self.dWh_l+=grad_left
        self.dWh_r += grad_right
        back_grad_left = np.dot(delta, self.Wh_l.T)
        back_grad_right = np.dot(delta, self.Wh_r.T)
        self.backward_tree(tree.subtrees[0], back_grad_left)
        self.backward_tree(tree.subtrees[1], back_grad_right)

    def backward(self, tree):
        """
        Back propagation
        Params:
            tree: root node
            softmax_layer: softmax layer of NN
            label: true label
        Returns:
            xxx
        """
        back_grad = self.backward_softmax(tree)
        self.backward_tree(tree,back_grad)

    def cost_grad(self, mnb_trees ,test=False):
        cost = 0.0

        mnb_predict = []
        mnb_correct = []

        # Zero gradients
        self.dWh_l[:] = 0
        self.dWh_r[:] = 0
        self.dbh[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0

        for i,tree in enumerate(mnb_trees):
            _cost,pred = self.forward(tree)
            cost += _cost
            mnb_predict.append(pred)
            mnb_correct.append(tree.label)

        if test is True:
            return (1./ len(mnb_trees)) * cost, mnb_correct, mnb_predict

        # backward
        for i,tree in enumerate(mnb_trees):
            self.backward(tree)

        avg = 1. / self.mnb_size

        cost += (self.l2_reg_level/2)*np.sum(self.Wh_l**2)
        cost += (self.l2_reg_level/2)*np.sum(self.Wh_r**2)
        cost += (self.l2_reg_level/2)*np.sum(self.Ws**2)

        cost = avg*cost
        # print("cost {0}".format(cost))

        gWh_l = avg * (self.dWh_l+ self.l2_reg_level*self.Wh_l)
        gWh_r = avg * (self.dWh_r+ self.l2_reg_level*self.Wh_r)
        gbh = avg * self.dbh
        gWs = avg * (self.dWs  + self.l2_reg_level*self.Ws)
        gbs = avg * self.dbs

        return cost, gWh_l, gWh_r, gbh, gWs, gbs
    def SGD(self, trees):

        np.random.shuffle(trees)

        n_samples = len(trees)
        costs = []

        for start_idx in range(0, n_samples, self.mnb_size):
            mnb_trees = trees[start_idx:start_idx+self.mnb_size]
            # update parameters
            cost, gWh_l, gWh_r, gbh, gWs, gbs = self.cost_grad(mnb_trees)
            self.Wh_l -= self.lr * gWh_l
            self.Wh_r -= self.lr * gWh_r
            self.bh -= self.lr * gbh
            self.Ws -= self.lr * gWs
            self.bs -= self.lr * gbs
            costs.append(cost)
        return costs
    def AdaGrad(self, trees, p=0.95, epsilon=1e-6):
        n_samples = len(trees)
        np.random.shuffle(trees)

        costs = []

        for start_idx in range(0, n_samples, self.mnb_size):
            mnb_trees = trees[start_idx:start_idx+self.mnb_size]

            # update parameters
            cost, gWh_l, gWh_r, gbh, gWs, gbs = self.cost_grad(mnb_trees)

            if not self.Wh_r_gt.any() or not self.Wh_l_gt.any():
                # t = t + G^2
                self.Wh_l_gt = gWh_l**2
                self.Wh_r_gt = gWh_r**2
                self.Ws_gt = gWs**2
                self.bh_gt = gbh**2
                self.bs_gt = gbs**2
            else:
                self.Wh_l_gt = p * self.Wh_l_gt + (1 - p) * gWh_l**2
                self.Wh_r_gt = p * self.Wh_r_gt + (1 - p) * gWh_r**2
                self.Ws_gt = p * self.Ws_gt + (1 - p) * gWs**2
                self.bh_gt = p * self.bh_gt + (1 - p) * gbh**2
                self.bs_gt = p * self.bs_gt + (1 - p) * gbs**2

            self.Wh_l -= self.lr * gWh_l / (np.sqrt(self.Wh_l_gt) + epsilon)
            self.Wh_r -= self.lr * gWh_r / (np.sqrt(self.Wh_r_gt) + epsilon)
            self.Ws -= self.lr * gWs / (np.sqrt(self.Ws_gt) + epsilon)
            self.bh -= self.lr * gbh / (np.sqrt(self.bh_gt) + epsilon)
            self.bs -= self.lr * gbs / (np.sqrt(self.bs_gt) + epsilon)

            costs.append(cost)

        return costs
    def train(self, X_train , X_valid, max_patience=20):

        epoch = 0

        train_acc = []
        dev_acc = [0]
        best_pos = 0
        patience = max_patience

        while True:
            start = time.time()
            # optimize function
            self.SGD(X_train)
            # self.SGD(trees)
            end = time.time()

            costt, correct, guess = self.cost_grad(X_train,test=True)
            t_acc = 0
            for i in xrange(0, len(correct)):
                t_acc += (guess[i] == correct[i])
            t_acc /= float(len(correct))
            train_acc.append(t_acc)

            costd, correct, guess = self.cost_grad(X_valid, test=True)
            d_acc = 0
            for i in xrange(0, len(correct)):
                d_acc += (guess[i] == correct[i])
            d_acc /= float(len(correct))
            dev_acc.append(d_acc)

            if d_acc < dev_acc[best_pos]:
                patience -= 1
            else:
                patience = max_patience
                best_pos = len(dev_acc) - 1
                with open('best_params.pickle', 'wb') as f:
                    pickle.dump((self.Wh_l,self.Wh_r,self.bh, self.Ws, self.bs), f)
            print "Epoch %d, %fs/epoch, train cost %.4f, val cost %.4f, train acc %.4f, val acc %.4f, patience %d"%(epoch, end-start, costt, costd, t_acc, d_acc, patience)

            if patience == 0:
                break
            epoch += 1
    def check_grad(self, data, epsilon= 1e-6):

        cost, gWh_l, gWh_r, gbh, gWs, gbs = self.cost_grad(data)
        err = 0.0
        count = 0.0

        print 'Checking dWh_l...'
        Wh_l = self.Wh_l[...,None]
        dWh_l = gWh_l
        for i in xrange(Wh_l.shape[0]):
            for j in xrange(Wh_l.shape[1]):
                Wh_l[i,j] += epsilon
                costP,_,_,_,_,_ = self.cost_grad(data)
                Wh_l[i,j] -= epsilon
                grad = (costP - cost) / epsilon
                err += np.abs(dWh_l[i, j] - grad)
                count+=1

        if 0.001 > err/count:
            print "Grad check passed for dWh_l"
        else:
            print "Grad check failed for dWh_l: sum of error = %.9f"%(err/count)

        err = 0.0
        count = 0.0
        print 'Checking dWh_r...'
        Wh_r = self.Wh_r[...,None]
        dWh_r = gWh_r
        for i in xrange(Wh_r.shape[0]):
            for j in xrange(Wh_r.shape[1]):
                Wh_r[i,j] += epsilon
                costP,_,_,_,_,_ = self.cost_grad(data)
                Wh_r[i,j] -= epsilon
                grad = (costP - cost) / epsilon
                err += np.abs(dWh_r[i, j] - grad)
                count+=1

        if 0.001 > err/count:
            print "Grad check passed for dWh_r"
        else:
            print "Grad check failed for dWh_r: sum of error = %.9f"%(err/count)


        err = 0.0
        count = 0.0
        print 'Checking dbh...'
        bh = self.bh[...,None]
        dbh = gbh
        for i in xrange(bh.shape[0]):
            for j in xrange(bh.shape[1]):
                bh[i,j] += epsilon
                costP,_,_,_,_,_ = self.cost_grad(data)
                bh[i,j] -= epsilon
                grad = (costP - cost) / epsilon
                err += np.abs(dbh[i, j] - grad)
                count+=1

        if 0.001 > err/count:
            print "Grad check passed for dbh"
        else:
            print "Grad check failed for dbh: sum of error = %.9f"%(err/count)


from tree import *
from vector.wordvectors import WordVectors
from preparedata import load_sentiment_data


def compute_cost_and_grad(theta, instances, total_internal_node_num,word_vectors, embsize, lambda_reg):

    """
    Params:
        theta: weight of model: Wh_l, Wh_r ...
        instances: all data
        word_vectors: wordvector for build RNN
        lambda_reg:
    Returns:
        total_cost: cost of foward
        total_grad: of theta
    """


    cost = 0
    grad = 0
    forward=instances
    return cost, grad


if __name__ == "__main__":

    rng = np.random.RandomState(4488)
    wordvector = WordVectors.load_from_text_format("../model/cwvector.txt", "cwvector")

    rnn = RecursiveNeuralNetworl(embsize=wordvector.embsize,mnb_size=30,lr=0.1,wordvector=wordvector,act_func=act.tanh)
    X_train, X_valid  = load_sentiment_data()

    rnn.train(X_train , X_valid)

    # with open("../data/rt-polarity.neg.out.txt", mode="r") as f:
    #     neg_trees_str = f.readlines()
    #
    # X_neg = []
    # for neg_tree_str in neg_trees_str[:5]:
    #     t = Tree(neg_tree_str)
    #     t = merge_bin_tree(t)
    #     t.label = 0
    #     X_neg.append(t)
    # print(len(X_neg))
    #
    # rnn.cost_grad(X_neg)
