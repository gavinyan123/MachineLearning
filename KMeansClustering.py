import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pandas as pd
import random

def random_centroids(df, K):
    # init_centroids = random.sample(range(0, len(df)),K)
    # centroids = []

    # for i in init_centroids:
    #     centroids.append(df.loc[i])
    # centroids = np.array(centroids)
    # return centroids
    N = df.shape[0]
    percentile_list  = [x for x in range(0,N,int(N/(K+1)))]
    return df.iloc[percentile_list[1:K+1]]

def calc_distance(X1, X2):
    return (sum((X1 - X2)**2))**0.5

# Assign cluster clusters based on closest centroid
def assign_clusters(centroids, cluster_array):
    clusters = []
    for i in range(cluster_array.shape[0]):
        distances = []
        for centroid in centroids:
            distances.append(calc_distance(centroid, 
                                           cluster_array[i]))
        cluster = [z for z, val in enumerate(distances) if val==min(distances)]
        clusters.append(cluster[0])
    return clusters

# Calculate new centroids based on each cluster's mean
def calc_centroids(clusters, cluster_array):
    new_centroids = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters, 
                                         columns=['cluster'])], 
                           axis=1)
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']\
                                     ==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

# Calculate variance within each cluster
def calc_centroid_variance(clusters, cluster_array):
    sum_squares = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters, 
                                         columns=['cluster'])], 
                           axis=1)
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']\
                                     ==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        mean_repmat = np.matlib.repmat(cluster_mean, 
                                       current_cluster.shape[0],1)
        sum_squares.append(np.sum(np.sum((current_cluster - mean_repmat)**2)))
    return sum_squares

# def findClosestCentroids(init, X):
#     assigned_centroid = []
#     for i in X: 
#         distance=[]
#         for j in init: 
#             distance.append(calc_distance(i,j))
#         assigned_centroid.append(np.argmin(distance))
#     return assigned_centroid

def plot_centroids(data, x, y, centroids):
    plt.scatter(data[x],data[y],c='black')
    plt.scatter(centroids[x],centroids[y],c='green')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def plot_clusters(df, x, y, centroids, cluster_array, df_centroids, K):
    clusters = assign_clusters(centroids, cluster_array)
    df['clusters'] = clusters
    for k in range(K):
        color = np.random.rand(1,3)
        # print(color)
        data=df[df['clusters']==k]
        plt.scatter(data[x],data[y],c=color)
        # print(data)
    plt.scatter(centroids2[x],centroids2[y],c='green')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

if __name__ == "__main__":
    # names = ['id_num','thickness','uni_size','uni_shape','adhesion','epi_size','bare_nuclei','chromatin','nucleoli','mitoses','class']
    names_n = ['id_num', 'outcome', 'rad', 'texture', 'perim', 'area', 'smooth', 'compact', 'concave', 'concave_points',
               'sym', 'fractal_dim', \
               'rad_SE', 'texture_SE', 'perim_SE', 'area_SE', 'smooth_SE', 'compact_SE', 'concave_SE',
               'concave_points_SE', 'sym_SE', 'fractal_dim_SE', \
               'rad_worst', 'texture_worst', 'perim_worst', 'area_worst', 'smooth_worst', 'compact_worst',
               'concave_worst', 'concave_points_worst', 'sym_worst', 'fractal_dim_worst']
    #file = '../wdbc.data'
    file = 'wdbc.data'

    df =pd.read_csv(file,index_col=False,header=None, names= names_n)
    df['outcome'] = df['outcome'].map(lambda diag: bool(diag == "M"))
    df.sort_values( by=['area'], inplace=True)

    cluster_data = df[['rad','compact']].copy(deep=True)
    cluster_data.dropna(axis=0, inplace=True)
    cluster_data.sort_values(by=['rad','compact'], inplace=True)
    cluster_array = np.array(cluster_data)
    # print(findClosestCentroids(centroids, df))
    k = 4
    cluster_vars = []
    centroids = [cluster_array[i+2] for i in range(k)]
    clusters = assign_clusters(centroids, cluster_array)
    initial_clusters = clusters
    cluster_data['clusters'] = clusters
    # print(0, round(np.mean(calc_centroid_variance(clusters, cluster_array))))
    for i in range(8):
        centroids = calc_centroids(clusters, cluster_array)
        clusters = assign_clusters(centroids, cluster_array)
        cluster_var = np.mean(calc_centroid_variance(clusters, 
                                                    cluster_array))
        cluster_vars.append(cluster_var)
        # print(i+1, round(cluster_var))
    x_label = []
    for i in range(len(cluster_vars)):
        x_label.append(i+1)

    plt.plot(x_label, cluster_vars,'go--', linewidth=1.5, markersize=4)
    plt.show()

    centroids2 = np.array(centroids)
    print(centroids2)
    centroids2 = pd.DataFrame(centroids2, columns = ['rad', 'compact'])
    print(centroids2)
    print(type(centroids2))

    plot_centroids(df, 'rad', 'compact', centroids2)
    plot_clusters(cluster_data, 'rad', 'compact', centroids, cluster_array, centroids2, k)