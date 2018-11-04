from src.main.config import features_path, clusters_path
from src.main.algorithms.utils import serialize, deserialize, convert_sparse_matrix_to_lists, manhattan_distance
from src.main.algorithms.kd_tree_seed.kd_tree_seed import KdTreeSeed
from src.main.algorithms.kmeans import KMeans


def perform_clustering(feature_path):
    features_mat = deserialize(features_path + '\\' + feature_path)
    features = convert_sparse_matrix_to_lists(features_mat)

    # get seeds
    kdtreeseed = KdTreeSeed(features, 9)
    first, second = kdtreeseed.get_seeds()
    kmeans = KMeans(features, 9, manhattan_distance, first)
    kmeans.fit()
    serialize(kmeans, clusters_path + '\\' + feature_path + '\\' + '.cluster')

perform_clustering('small.csv.features')


