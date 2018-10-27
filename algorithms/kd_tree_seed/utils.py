
def get_bounding_points(points):
    nr_feat = len(points[0])
    min_pt, max_pt = [None] * nr_feat, [None] * nr_feat
    for point in points:
        for feature_idx in range(nr_feat):
            if min_pt[feature_idx] is None or point[feature_idx] < min_pt[feature_idx]:
                min_pt[feature_idx] = point[feature_idx]
            if max_pt[feature_idx] is None or point[feature_idx] > max_pt[feature_idx]:
                max_pt[feature_idx] = point[feature_idx]
    return min_pt, max_pt
