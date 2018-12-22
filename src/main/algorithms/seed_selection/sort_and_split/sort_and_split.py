from src.main.algorithms.utils import euclidean_distance
import copy


class SortAndSplit:
    def __init__(self, points, k):
        self.points = points
        self.k = k
        self.normalized_pts = copy.deepcopy(points)

    def get_seeds(self):
        """
        Gets k seeds from given points using the sort and split heuristic
        :return: the means detected by the algorithm
        """
        min_features = self.get_min_feature_values()
        marked_features = self.get_features_to_normalize()
        self.normalize_negative_points(min_features, marked_features)
        self.sort_points()
        return self.get_seeds_from_partition()

    def normalize_negative_points(self, min_features, marked_features):
        """
        Normalizes the values of the features by subtracting the min value of a feature
        from each point if there is a point with a negative value for that feature
        :return:
        """
        for point_idx in range(len(self.points)):
            point = self.normalized_pts[point_idx]
            for feature_idx in range(len(point)):
                if marked_features[feature_idx] == 1:
                    point[feature_idx] -= min_features[feature_idx]
            self.normalized_pts[point_idx] = point

    def get_features_to_normalize(self):
        """
        Gets an array the size of the nr of features which marks those who need to be normalized
        1 - there is a point with a negative value for this feature
        0 - no points with negative values for this feature
        :return:
        """
        feature_size = len(self.points[0])
        marked_features = [0] * feature_size
        for point in self.points:
            for feature_idx in range(feature_size):
                if point[feature_idx] < 0:
                    marked_features[feature_idx] = 1
        return marked_features

    def get_min_feature_values(self):
        """
        Gets an array with the minimum values for each feature
        :return:
        """
        min_features = self.points[0]
        for point in self.points:
            for feature_idx in range(len(min_features)):
                if point[feature_idx] < min_features[feature_idx]:
                    min_features[feature_idx] = point[feature_idx]
        return min_features

    def calculate_distances_from_origin(self):
        distances = []
        origin = [0] * len(self.points[0])
        for point in self.normalized_pts:
            distances.append(euclidean_distance(point, origin))
        return distances

    def sort_points(self):
        """
        Sorts all points by distance from origin
        :return:
        """
        distances = self.calculate_distances_from_origin()
        for i in range(len(self.points) - 1):
            for j in range(i + 1, len(self.points)):
                if distances[i] > distances[j]:
                    distances[i], distances[j] = distances[j], distances[i]
                    self.points[i], self.points[j] = self.points[j], self.points[i]
                    self.normalized_pts[i], self.normalized_pts[j] = self.normalized_pts[j], self.normalized_pts[i]

    def get_seeds_from_partition(self):
        """
        Gets an array of seeds from the partition dataset
        :return:
        """
        seeds = []
        partition_size = int(len(self.points) / self.k)
        lower_bound = 0
        upper_bound = partition_size
        middle = int(partition_size / 2) + lower_bound
        while len(seeds) < self.k:
            seeds.append(self.points[middle])
            lower_bound = upper_bound
            upper_bound += partition_size
            middle = int(partition_size / 2) + lower_bound
        return seeds
