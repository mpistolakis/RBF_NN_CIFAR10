"""
The current file contains the RBF neuron Class, it is responsible for the initialization of Sigma and centers,
if they will be trainable and the implementation of mathematical expression.
"""

from keras import backend as K
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
import pickle
from keras.constraints import maxnorm, min_max_norm
from tensorflow.keras.layers import Layer


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas_init=None, betas=2.0, trainable_centers=True,
                 trainable_betas=True, **kwargs):
        self.betas_init = betas_init
        self.output_dim = output_dim
        self.initializer = initializer
        self.init_betas = betas
        self.trainable_centers = trainable_centers
        self.trainable_betas = trainable_betas
        if betas == 2.0:
            self.init_betas = self.betas_init
        else:
            self.init_betas = RandomUniform(8, 15)

        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape[0], input_shape[1], 'shape')
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=self.trainable_centers)

        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=self.init_betas,
                                     trainable=self.trainable_betas,
                                     )

        super(RBFLayer, self).build(input_shape)

    def get_betas(self):
        return self.betas

    def call(self, x):
        # c = K.expand_dims(self.centers)
        # H = K.transpose(c - K.transpose(x))
        # return K.exp(K.sum(H ** 2, axis=1) / -2 * self.betas)  # -self.betas**2

        c = self.centers[np.newaxis, :, :]
        x = x[:, np.newaxis, :]

        diffnorm = K.sum((c - x) ** 2, axis=-1)
        ret = K.exp(- self.betas * diffnorm)

        # ret = K.exp(- diffnorm / (self.betas ** 2) * 2)  # σ^2 ειναι
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
