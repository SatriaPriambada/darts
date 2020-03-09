import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np

from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
np.random.seed(1)

def draw_graph(df):
    print(df)
    clusters = 8

    lat_data = df[['mean_lat', 'lat95', 'lat99']].copy()
    print(lat_data)
    np_data = lat_data.to_numpy()
    print(np_data)

    # Obtain labels for each point in mesh. Pick kmedioid because more robust
    # against outlier. In case of latency there will be outlier
    # Medioid also pi 
    # Use euclidean matrix because all is lat in ms
    # in case of future grouping to add more data modality
    # columns to consider (e.g FLOP, etc.) change metric to Manhattan
    
    kmedoids = KMedoids(metric="euclidean", n_clusters=clusters, random_state=0).fit(np_data)
    label = kmedoids.labels_
    df['label'] = label
    print(df)
    center = kmedoids.cluster_centers_
    print(center)
    centers_genotype = []
    for i in range(clusters):
        print("center x{}, y{}, z{}".format(center[i][0],center[i][1],center[i][2]))
        centers_genotype.append(df[(df['mean_lat']==center[i][0]) & (df['lat95']==center[i][1]) & (df['lat99']==center[i][2]) ])

    print(centers_genotype)
    
    #reduced_data = PCA(n_components=2).fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For each set of mean, lat95 and lat 99 draw each point color based on kmedioid label
    color_choice = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for index, row in df.iterrows():
        xs = row['mean_lat']
        ys = row['lat95']
        zs = row['lat99']
        color = color_choice[row['label']]
        ax.scatter(xs, ys, zs, c=color)

    ax.set_xlabel('mean_lat')
    ax.set_ylabel('tail_lat_99%')
    ax.set_zlabel('tail_lat_95%')
    labels = []
    for cluster_id in range(8):
        labels.append(Patch(facecolor=color_choice[cluster_id],
                            label='KMedioid Cluster {}'.format(cluster_id)))
    lgd = ax.legend(handles=labels)
    f = ax.get_figure()
    f.savefig('3D_lat_kmedioid.pdf', ext='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    csv_path = 'latencies_sampled_genotypes.csv'
    df = pd.read_csv(csv_path)
    draw_graph(df) 