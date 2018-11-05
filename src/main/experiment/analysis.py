from src.main.algorithms.utils import deserialize
from src.main.config import features_path, clusters_path, processed_path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def do_analysis(clusters_file, labels_file, words_file, data_file):
    clusters = deserialize(clusters_path + "\\" + clusters_file)
    labels = deserialize(clusters_path + "\\" + labels_file)
    words = deserialize(features_path + "\\" + words_file)
    beer_df = pd.read_csv(processed_path + "\\" + data_file)

    find_representative_words(clusters, words)
    make_heatmap(labels, beer_df)


def find_representative_words(clusters, words):
    """
    Prints the most representative words for each cluster
    Sorts the features of the cluster mean and takes the first 10
    """
    words_per_cluster = []
    for cluster in clusters:
        temp = [(v, i) for i, v in enumerate(cluster)]
        temp.sort(reverse=True)
        cluster_words = []
        for idx in range(10):
            v, i = temp[idx]
            cluster_words.append(words[i])
        words_per_cluster.append(cluster_words)

    for cluster_idx in range(len(words_per_cluster)):
        print(cluster_idx, " : ", words_per_cluster[cluster_idx])


def make_heatmap(labels, beer_df):
    # add a new column ; pointing each beer to its label
    beer_df['cluster_id'] = labels

    # creates a mapping between cluster - beer_style - apparitions of beer style in cluster
    clusters = beer_df.groupby(['cluster_id', 'beer_style']).size()

    fig, ax = plt.subplots()
    # first col = cluster, following cols are values for beer styles
    sns.heatmap(clusters.unstack(level='beer_style'), ax=ax, cmap='Reds')
    ax.set_ylabel('cluster_id', fontdict={'weight': 'bold', 'size': 26})
    plt.xticks(rotation=30)
    plt.show()


do_analysis('small.csv.features.centers.kd',
            'small.csv.features.labels.kd',
            'small.csv.feature_names',
            'small.csv')

