
""" In this file there are methods for initializing the Centers in such ways: randomly, with K-nearest centroid and
with K-means algorithms. Also here are initialized the possibles optimal Sigmas bases of data structure and centers. """

import math
import pickle
import numpy as np
# from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
import pickle



def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def install():
    data1 = unpickle(r"cifar-10-batches-py\data_batch_1")
    data2 = unpickle(r"cifar-10-batches-py\data_batch_2")
    data3 = unpickle(r"cifar-10-batches-py\data_batch_3")
    data4 = unpickle(r"cifar-10-batches-py\data_batch_4")
    data5 = unpickle(r"cifar-10-batches-py\data_batch_5")
    trainingLabels = np.array(
        data1[b'labels'] + data2[b'labels'] + data3[b'labels'] + data4[b'labels'] + data5[b'labels'])
    return trainingLabels


class InitCentersRandom(Initializer):
    """ Initializer """

    def __init__(self, X, n_centers_of_each_class):
        self.X = X
        self.n_centers_of_each_class = n_centers_of_each_class

    def __call__(self, shape, dtype=None):
        # assert shape[1] == self.X.shape[1]
        train_labels = install()
        idx = []

        for i in range(10):
            label_index = np.where(train_labels == i)
            label_that_we_want = label_index[0]
            print(label_that_we_want)
            for k in range(self.n_centers_of_each_class):
                flag = True
                while flag:
                    label = np.random.choice(label_that_we_want)
                    if label not in idx:
                        flag = False
                        idx.append(label)
        idx = np.array(idx)
        # idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


""" This function find the optimal sigma of each center and the corresponding points"""


class betas_for_k_nearest_centroid(Initializer):
    def __init__(self, centers, name, X, Y):
        self.centers = centers
        self.name = name
        self.X = X
        self.Y = Y

    def __call__(self, shape, dtype=None):
        diaspora = []
        for label in self.name:
            a_list = []
            sums = 0
            idx = np.where(self.Y == label)
            idx = idx[0]
            for i in idx:
                a_list.append(self.X[i])
            data = np.array(a_list)
            for k in data:
                sums = np.linalg.norm(k - self.centers[label]) + sums
            sums = sums / 5000.0
            diaspora.append(sums)

        diaspora = np.array(diaspora)
        return diaspora


""" This function find the centers of each class """


class InitCentersNearestCentroid(Initializer):
    def __init__(self, X, Y, centers=None, names=None):
        self.X = X
        self.Y = Y
        self.centers = centers
        self.names = names
        self.nearest_compute()

    def __call__(self, shape, dtype=None, *args, **kwargs):
        return self.centers

    def nearest_compute(self):
        centroid_classifier = NearestCentroid(metric="manhattan")
        centroid_classifier.fit(self.X, self.Y)
        centers = centroid_classifier.centroids_
        self.names = centroid_classifier.classes_
        self.centers = centers


""" Centers Initializer with K-means clusters   """


class InitCentersKMeans(Initializer):

    def __init__(self, X, clusters=None, names=None, n_centers=None, max_iter=300):
        self.X = X
        self.max_iter = max_iter
        self.clusters = clusters
        self.n_centers = n_centers
        self.names = names
        self.K_means_compute()

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        return self.clusters

    def K_means_compute(self):
        km = KMeans(n_clusters=self.n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        clusters = km.cluster_centers_
        names = km.labels_
        self.clusters = clusters
        self.names = names


class k_means_find_optimal_sigma(Initializer):

    def __init__(self, centers, X, names):
        self.centers = centers
        self.X = X
        self.names = names

    def __call__(self, shape, dtype=None):
        diaspora = []
        for m in range(len(self.centers)):
            a_list = []
            sums = 0
            # train=install()
            # idx=ClusterIndicesNumpy(center,train)
            idx = np.where(self.names == m)
            idx = idx[0]
            # print(len(idx), 'idx')
            for i in idx:
                a_list.append(self.X[i])
            data = np.array(a_list)
            number = len(data)

            for k in data:
                sums = np.linalg.norm(k - self.centers[m]) + sums
            sums = sums / number
            if sums != 0:
                sums = 1 / math.pow(sums, 2)
            else:
                sums = 1
            diaspora.append(sums)
        diaspora = np.array(diaspora)
        return diaspora
