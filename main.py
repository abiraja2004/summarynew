import theano
import theano.tensor as T
import numpy as np
if __name__ == "__main__":

    rng = np.random.RandomState(4488)

    X_ = rng.uniform(size=(3,2*5*5))
    print (X_)
    X = T.dmatrix("X")

    hy = T.reshape(X,(3,2,5,5))

    bias = theano.shared(value= np.asarray(rng.uniform(-1,1,(2,5,5)) ,dtype=np.float64),
                                   name ="b",
                                   borrow=True
            )

    bias_ = bias.get_value()

    AA = hy + bias
    print bias.shape

    show_function = theano.function([X], [hy,AA])
    hy_,AA_ = show_function(X_)
    print(AA_.shape)