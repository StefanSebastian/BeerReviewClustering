from functools import reduce

from src.main.algorithms.utils import euclidean_distance

from src.main.algorithms.kd_tree_seed.utils import get_bounding_points


class LeafBucket:
    def __init__(self, points):
        self.points = points
        self.nr_elem = len(points)
        self.nr_feat = len(points[0])

        self.volume = None
        self.density = None
        self.mean = None

        self.calculate_stats()

    def calculate_stats(self):
        """
        Calculates all descriptive information about this bucket : volume, density, mean
        """
        self.calculate_volume()
        self.calculate_density()
        self.calculate_mean()

    def calculate_volume(self):
        min_pt, max_pt = get_bounding_points(self.points)
        dimensions = [0] * self.nr_feat
        non_zero = []

        for feature_idx in range(self.nr_feat):
            dimension = max_pt[feature_idx] - min_pt[feature_idx]
            if dimension != 0:
                dimensions[feature_idx] = dimension
                non_zero.append(dimension)

        geometric_mean = reduce(lambda x, y: x * y, non_zero) ** (1.0 / len(non_zero))

        volume = 1
        for dimension in dimensions:
            if dimension == 0:
                volume *= geometric_mean
            else:
                volume *= dimension

        if volume == 0:  # handle underflow
            scale = 1 / geometric_mean  # try to get mean towards 1
            volume = 1
            for dimension in dimensions:
                if dimension == 0:
                    volume = volume * geometric_mean * scale
                else:
                    volume = volume * dimension * scale

        self.volume = volume

    def calculate_density(self):
        self.density = len(self.points) / self.volume

    def calculate_mean(self):
        mean = [0] * self.nr_feat
        for elem in self.points:
            for feature_idx in range(self.nr_feat):
                mean[feature_idx] += elem[feature_idx]
        for feature_idx in range(self.nr_feat):
            mean[feature_idx] /= self.nr_elem
        self.mean = mean


class CenterSelector:
    def __init__(self, leaf_buckets, k):
        self.leaf_buckets = leaf_buckets
        self.k = k

    def get_seeds(self):
        self.leaf_buckets.sort(key=lambda x: x.density, reverse=True)

        first_seed = self.seed_extraction()
        drop_count = int((1 / 5) * len(self.leaf_buckets))
        self.leaf_buckets = self.leaf_buckets[:len(self.leaf_buckets) - drop_count]
        second_seed = self.seed_extraction()

        return self.convert_to_means_list(first_seed), \
               self.convert_to_means_list(second_seed)

    @staticmethod
    def convert_to_means_list(leaf_buckets):
        res = []
        for bucket in leaf_buckets:
            res.append(bucket.mean)
        return res

    def seed_extraction(self):
        selected = [self.leaf_buckets[0]]
        # first selection is the one with most density

        for current_seed_idx in range(1, self.k):
            max_weight, max_weight_idx = -1, -1
            for considered_bucket_idx in range(len(self.leaf_buckets)):
                weight = self.calculate_mean_weight(self.leaf_buckets[considered_bucket_idx], selected)
                if max_weight == -1 or max_weight < weight:
                    max_weight = weight
                    max_weight_idx = considered_bucket_idx
            selected.append(self.leaf_buckets[max_weight_idx])
        return selected

    @staticmethod
    def calculate_mean_weight(considered_bucket, selected_seeds):
        """
        Get the product of the considered bucked density and the min distance between the bucket and a selected seed
        gj = {min k=1..t[d(ck, mj)]} * pj

        :param considered_bucket: considered bucket
        :param selected_seeds: currently selected seeds
        :return:
        """
        min_dist = -1
        for selected_seed in selected_seeds:
            distance = euclidean_distance(considered_bucket.mean, selected_seed.mean)
            if distance < min_dist or min_dist == -1:
                min_dist = distance
        return min_dist * considered_bucket.density
