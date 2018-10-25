from collections import namedtuple
from operator import itemgetter
from pprint import pformat


class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))


def kdtree(point_list, depth=0):
    try:
        k = len(point_list[0])
    except IndexError as e:
        return None

    axis = depth % k

    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2

    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1)
    )


def main():
    point_list = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = kdtree(point_list)
    print(tree)


if __name__ == '__main__':
    main()
