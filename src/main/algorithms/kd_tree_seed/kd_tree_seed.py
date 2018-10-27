from functools import reduce

import math

from src.main.algorithms.kd_tree_seed.kd_tree import KdTree

from src.main.algorithms.kd_tree_seed.build_candidates import LeafBucket, CenterSelector
from src.main.algorithms.kd_tree_seed.utils import get_bounding_points


class KdTreeSeed:
    def __init__(self, points, k):
        self.points = points
        self.k = k

    def get_seeds(self):
        """
        Gets k seeds from given points using kd-tree algorithm
        :param k: the number of desired seeds
        :return: the means detected by the algorithm
        """
        kdtree = KdTree(self.points,
                        len(self.points) // (10 * self.k))  # to get approximately 10 leaf buckets for each seed
        kdtree.build()

        leaf_buckets = []
        scale = self.get_volume_scale(kdtree.leaves)
        for leaf_points in kdtree.leaves:
            leaf_buckets.append(LeafBucket(leaf_points, scale))

        selector = CenterSelector(leaf_buckets, self.k)
        return selector.get_seeds()

    @staticmethod
    def get_volume_scale(leaves):
        """
        For some datasets with a large nr of features a scaling factor must be applied
        when calculating volume
        """
        scale = lambda x: x  # no scale by default
        for leaf_points in leaves:
            non_zero = []
            min_pt, max_pt = get_bounding_points(leaf_points)
            for feature_idx in range(len(leaf_points[0])):
                dimension = max_pt[feature_idx] - min_pt[feature_idx]
                if dimension != 0:
                    non_zero.append(dimension)
            geometric_mean = reduce(lambda x, y: x * y, non_zero) ** (1.0 / len(non_zero))

            if geometric_mean * len(leaf_points[0]) == 0:
                # underflow
                scale = lambda x: math.exp(x)
        return scale
