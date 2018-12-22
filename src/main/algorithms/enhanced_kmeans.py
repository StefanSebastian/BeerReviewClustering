import datetime # logging

from src.main.algorithms.seed_selection.random.random_seeds import RandomSeeds
from src.main.algorithms.utils import manhattan_distance


class EnhancedKMeans:
    def __init__(self, data, k, distance_function=manhattan_distance, means=None):
        self.data = data
        self.data_size = len(data)
        self.feature_size = len(data[0])
        self.k = k
        self.means = means
        self.distance_function = distance_function
        self.labels = [-1] * self.data_size
        self.assignment_change = True
        self.prev_dist = []

    def select_random_means(self):
        """
        Default seed selection strategy
        :return:
        """
        random_seeds = RandomSeeds(self.data, self.k)
        self.means = random_seeds.get_seeds()

    def mean_assignment(self):
        """
        Updates the point assignments
        If the distance to the updated mean is smaller than what was stored
        then that point is left in its cluster
        :return:
        """
        for assignee_idx in range(self.data_size):
            dist_to_mean = self.distance_function(self.data[assignee_idx], self.means[self.labels[assignee_idx]])
            if dist_to_mean > self.prev_dist[assignee_idx]:
                nearest_mean_idx = self.get_nearest_mean(assignee_idx)
                if nearest_mean_idx != self.labels[assignee_idx]:
                    self.assignment_change = True
                self.labels[assignee_idx] = nearest_mean_idx
                new_dist = self.distance_function(self.data[assignee_idx], self.means[nearest_mean_idx])
                self.prev_dist[assignee_idx] = new_dist

    def get_nearest_mean(self, assignee_idx):
        """
        Finds the nearest mean for the given point
        """
        best_distance = -1
        nearest_mean_idx = -1
        for mean_idx in range(self.k):
            distance = self.distance_function(self.data[assignee_idx], self.means[mean_idx])
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
                nearest_mean_idx = mean_idx
        return nearest_mean_idx

    def update_means(self):
        """
        Update mean values
        """
        sums_per_cluster = [[0 for _ in range(self.feature_size)] for _ in range(self.k)]
        elements_per_cluster = [0] * self.k

        for element_idx in range(self.data_size):
            element = self.data[element_idx]
            cluster_idx = self.labels[element_idx]

            for feature_idx in range(self.feature_size):
                sums_per_cluster[cluster_idx][feature_idx] += element[feature_idx]
            elements_per_cluster[cluster_idx] += 1

        for cluster_idx in range(self.k):
            nr_elems = elements_per_cluster[cluster_idx]
            for feature_idx in range(self.feature_size):
                if nr_elems != 0:
                    self.means[cluster_idx][feature_idx] = sums_per_cluster[cluster_idx][feature_idx] / nr_elems

    def compute_auxiliary_distance_structure(self):
        """
        Computes the auxiliary structure that contains the distance from each point to its nearest mean
        :return:
        """
        for assignee_idx in range(self.data_size):
            nearest_mean_idx = self.get_nearest_mean(assignee_idx)
            distance = self.distance_function(self.data[assignee_idx], self.means[nearest_mean_idx])
            self.prev_dist.append(distance)

    def fit(self):
        print('Started at : ', datetime.datetime.now())

        if self.means is None:
            print("No seed provided. Selecting random neans")
            self.select_random_means()

        # initial assignment
        self.compute_auxiliary_distance_structure()

        nr_iteration = 0
        while self.assignment_change:
            nr_iteration += 1
            print("Iteration : ", nr_iteration)
            self.assignment_change = False
            self.mean_assignment()
            self.update_means()

        print(self.means)
        print(self.labels)
        print('Finished at : ', datetime.datetime.now())


