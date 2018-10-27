from operator import itemgetter

from algorithms.kd_tree_seed.utils import get_bounding_points


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
    def __init__(self, points, max_leaf_size=1):
        self.leaves = []
        self.max_leaf_size = max_leaf_size
        self.points = points

    def build(self):
        return self.recursive_kdtree(self.points, 0)

    def recursive_kdtree(self, points_at_level, level):
        print('level: ', level, ' len points: ', len(points_at_level))
        if len(points_at_level) <= self.max_leaf_size:
            self.leaves.append(points_at_level)
            return Node(root=points_at_level, left_child=None, right_child=None)

        min_pt, max_pt = get_bounding_points(points_at_level)
        axis = self.get_axis(min_pt, max_pt)

        points_at_level.sort(key=itemgetter(axis))
        median = len(points_at_level) // 2
        return Node(
            root=points_at_level[median],
            left_child=self.recursive_kdtree(points_at_level[:median], level + 1),
            right_child=self.recursive_kdtree(points_at_level[median + 1:], level + 1)
        )

    def get_axis(self, min_pt, max_pt):
        nr_feat = len(self.points[0])
        longest_val, longest_idx = 0, 0
        for feature_idx in range(nr_feat):
            current_val = max_pt[feature_idx] - min_pt[feature_idx]
            if current_val >= longest_val:
                longest_val, longest_idx = current_val, feature_idx
        return longest_idx


def main():
    point_list = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = KdTree(point_list, 2)
    res = tree.build()
    res.print()


if __name__ == '__main__':
    main()
