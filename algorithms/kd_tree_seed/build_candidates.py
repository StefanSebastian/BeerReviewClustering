from functools import reduce

from algorithms.kd_tree_seed.utils import get_bounding_points


class LeafBucket:
    def __init__(self, points):
        self.points = points
        self.nr_elem = len(points)
        self.nr_feat = len(points[0])

        self.volume = None
        self.density = None

    def calculate_volume(self):
        min_pt, max_pt = get_bounding_points(self.points)
        dimensions = [0] * self.nr_feat
        non_zero = []

        for feature_idx in range(self.nr_feat):
            dimension = max_pt[feature_idx] - min_pt[feature_idx]
            if dimension != 0:
                dimensions[feature_idx] = dimension
                non_zero.append(dimension)

        geometric_mean = reduce(lambda x, y: x*y, non_zero)**(1.0/len(non_zero))

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
