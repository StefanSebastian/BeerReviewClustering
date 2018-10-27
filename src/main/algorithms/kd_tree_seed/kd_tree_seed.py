from src.main.algorithms.kd_tree_seed.kd_tree import KdTree

from src.main.algorithms.kd_tree_seed.build_candidates import LeafBucket, CenterSelector


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
        kdtree = KdTree(self.points, len(self.points) // (10 * self.k))  # to get approximately 10 leaf buckets for each seed
        kdtree.build()
        leaf_buckets = []
        for leaf_points in kdtree.leaves:
            leaf_buckets.append(LeafBucket(leaf_points))
        selector = CenterSelector(leaf_buckets, self.k)
        return selector.get_seeds()
