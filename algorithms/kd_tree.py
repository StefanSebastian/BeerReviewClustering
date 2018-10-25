from operator import itemgetter

from algorithms.utils import Data


class Node:
    def __init__(self, root, left_child, right_child):
        self.root = root
        self.left_child = left_child
        self.right_child = right_child

    def print(self):
        self.print_rec("", True)

    def print_rec(self, prefix, is_tail):
        print(prefix + ("'--" if is_tail else "|--") + str(len(self.root)))
        if self.left_child is not None:
            self.left_child.print_rec(prefix + ("  " if is_tail else "|  "), False)
        if self.right_child is not None:
            self.right_child.print_rec(prefix + ("  " if is_tail else "|  "), True)


class KdTree:
    def __init__(self, data, max_leaf_size=1):
        self.leaves = []
        self.max_leaf_size = max_leaf_size
        self.data = data

    def build(self):
        return self.recursive_kdtree(self.data.points, 0)

    def recursive_kdtree(self, points, level):
        print('level: ', level, ' len points: ', len(points))
        if len(points) <= self.max_leaf_size:
            self.leaves.append(points)
            return Node(root=points, left_child=None, right_child=None)

        min_pt, max_pt = self.get_bounding_points(points)
        axis = self.get_axis(min_pt, max_pt)

        points.sort(key=itemgetter(axis))
        median = len(points) // 2
        return Node(
            root=points[median],
            left_child=self.recursive_kdtree(points[:median], level + 1),
            right_child=self.recursive_kdtree(points[median + 1:], level + 1)
        )

    def get_bounding_points(self, points):
        nr_feat = self.data.nr_features
        min_pt, max_pt = [None] * nr_feat, [None] * nr_feat
        for point in points:
            for feature_idx in range(nr_feat):
                if min_pt[feature_idx] is None or point[feature_idx] < min_pt[feature_idx]:
                    min_pt[feature_idx] = point[feature_idx]
                if max_pt[feature_idx] is None or point[feature_idx] > max_pt[feature_idx]:
                    max_pt[feature_idx] = point[feature_idx]
        return min_pt, max_pt

    def get_axis(self, min_pt, max_pt):
        nr_feat = self.data.nr_features
        longest_val, longest_idx = 0, 0
        for feature_idx in range(nr_feat):
            current_val = max_pt[feature_idx] - min_pt[feature_idx]
            if current_val >= longest_val:
                longest_val, longest_idx = current_val, feature_idx
        return longest_idx


def main():
    point_list = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    data = Data(point_list, len(point_list), 2)
    tree = KdTree(data, 2)
    res = tree.recursive_kdtree(point_list, 1)
    res.print()


if __name__ == '__main__':
    main()
