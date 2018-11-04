import pickle

import math


def serialize(obj, filename):
    with open(filename, 'wb') as output_file:
        pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)


def deserialize(filename):
    with open(filename, 'rb') as input_file:
        return pickle.load(input_file)


def manhattan_distance(x, y):
    dist = 0
    for i in range(len(x)):
        dist += abs(x[i] - y[i])
    return dist


def euclidean_distance(x, y):
    dist = 0
    for i in range(len(x)):
        dist += (x[i] - y[i]) ** 2
    return math.sqrt(dist)


def convert_sparse_matrix_to_lists(matrix):
    temp_list = []
    for i in matrix:
        temp_list.append(list(i.A[0]))
    return temp_list
