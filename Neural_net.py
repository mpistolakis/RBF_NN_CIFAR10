"""The main script that handles the neural network architecture, the data preprocessing, the model training and
displays results with plots"""
import math
import tensorflow as tf
from tensorflow import keras
from keras.layers.core import Dense, Dropout
from tensorflow.python.keras.callbacks import LearningRateScheduler
import numpy as np
import winsound
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import time
from Tf_initializers import betas_for_k_nearest_centroid, InitCentersKMeans, InitCentersNearestCentroid, \
    k_means_find_optimal_sigma, InitCentersRandom
from RBF_Layer import RBFLayer

from keras.models import Model

tf.get_logger().setLevel('INFO')
""" Open the file for reading """


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


if __name__ == "__main__":
    data1 = unpickle(r"cifar-10-batches-py\data_batch_1")
    data2 = unpickle(r"cifar-10-batches-py\data_batch_2")
    data3 = unpickle(r"cifar-10-batches-py\data_batch_3")
    data4 = unpickle(r"cifar-10-batches-py\data_batch_4")
    data5 = unpickle(r"cifar-10-batches-py\data_batch_5")
    test = unpickle(r"cifar-10-batches-py\test_batch")
    label_names = unpickle(r"cifar-10-batches-py\batches.meta")

    """ collect all the training data in one np array """
    process1 = np.append(data1[b'data'], data2[b'data'], axis=0)
    process2 = np.append(process1, data3[b'data'], axis=0)
    process3 = np.append(process2, data4[b'data'], axis=0)
    trainingData = np.append(process3, data5[b'data'], axis=0)
    """ collect all the labels in one list """
    trainingLabels = np.array(
        data1[b'labels'] + data2[b'labels'] + data3[b'labels'] + data4[b'labels'] + data5[b'labels'])
    testLabels = np.array(test[b'labels'])
    testData = test[b'data']

    x_train = trainingData / 255.0
    x_test = testData / 255.0


    def plot_results(history, number_model):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy' + number_model)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        loss(history, number_model)


    def loss(history, number_model):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss ' + number_model)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


    pca = PCA(n_components=500)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    # scale the data with the max value

    maxt = max(x_train.flatten())
    maxtest = max(x_test.flatten())
    x_test = x_test / maxtest
    x_train = x_train / maxt
    # scale the data with min max formula

    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler.fit(x_train)
    # scaler.transform(x_train)
    # scaler.transform(x_test)
    # print(min(x_train.flatten()), min(x_test.flatten()), max(x_train.flatten()), max(x_test.flatten()))

    """in these function we decay the learning rate in specific time of epochs """


    def step_decay(epoch):
        initial_l_rate = 0.005
        drop = 0.5
        epochs_drop = 15
        l_rate = initial_l_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        print(l_rate, 'rate')
        return l_rate


    """Labels to vector"""

    y_test = testLabels
    y_train = trainingLabels
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)

    """Nearest centroid initializer"""

    # centers = InitCentersNearestCentroid(x_train, trainingLabels)

    """ K-means clusters Initializer """
    kmeans_centers = InitCentersKMeans(X=x_train, n_centers=50)

    """Model structure"""

    input_shape = x_train.shape[1]
    visible = keras.layers.Input(shape=(input_shape,))

    """ Here we build the RBF layer """
    rbf_layer = RBFLayer(output_dim=50,
                         trainable_centers=False,
                         # initializer=InitCentersRandom(x_train, n_centers_of_each_class=5),

                         # code to run nearest centroid with only 10 units initializer=centers,
                         # betas_init=betas_for_k_nearest_centroid(centers.centers, centers.names, centers.X,
                         # centers.Y),

                         # code to run kmeans initialization
                         initializer=kmeans_centers,
                         betas_init=k_means_find_optimal_sigma(kmeans_centers.clusters, kmeans_centers.X,
                                                               kmeans_centers.names),

                         # if betas==2.0 initializer= custom initializer  , else is random uniform
                         betas=2.1,
                         input_shape=(input_shape,))

    x = rbf_layer(visible)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=visible, outputs=output)
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=adam)

    start_time = time.time()
    print(rbf_layer.betas)

    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]
    print(callbacks_list, "list")


    # print(model.summary())

    def train(epochs, batch_size):
        model_history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                                  epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True,

                                  # callbacks=callbacks_list
                                  )

        return model_history


    plot_results(train(30, 10), "30 epoch 10 bath")
    print(rbf_layer.betas)


    def show_metrics():
        print("train accuracy "), model.evaluate(x_train, y_train)
        print("test accuracy "), model.evaluate(x_test, y_test)


    show_metrics()
    end_time = time.time()

    print("Time spent: (min) ", (end_time - start_time) / 60)

""" alarm when code finish"""
# duration = 1000
# freq = 200
# winsound.Beep(freq, duration)
