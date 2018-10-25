import pickle


def serialize(obj, filename):
    with open(filename, 'wb') as output_file:
        pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)


def deserialize(filename):
    with open(filename, 'rb') as input_file:
        return pickle.load(input_file)


class Data:
    def __init__(self, points, nr_points, nr_features):
        self.points = points
        self.nr_points = nr_points
        self.nr_features = nr_features
