#import libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# Select random observation as centroids
def plot_init_centroids(data, K):
    N = data.shape[0]
    percentile_list  = [x for x in range(0,N,int(N/(K+1)))]    
    Centroids = data.iloc[percentile_list[1:K+1]]
    # Centroids = (data.sample(n=K))
    # plt.scatter(data["rad"],X["compact"],c='black')
    # plt.scatter(Centroids["rad"],Centroids["compact"],c='red')
    # plt.xlabel('rad')
    # plt.ylabel('compact')
    # plt.show()
    return Centroids


def clustering(data, col1, col2, K):
    diff = 1
    j=0
    Centroids = plot_init_centroids(data, K)
    X = data
    elbow = []
    total = 0
    while(diff!=0):
        XD=X
        i=1
        for index1,row_c in Centroids.iterrows():
            ED=[]
            for index2,row_d in XD.iterrows():
                d1=(row_c[col1]-row_d[col1])**2
                d2=(row_c[col2]-row_d[col2])**2
                d=np.sqrt(d1+d2)
                ED.append(d)
            X[i]=ED
            i=i+1

        C=[]
        for index,row in X.iterrows():
            min_dist=row[1]
            pos=1
            for i in range(K):
                if row[i+1] < min_dist:
                    min_dist = row[i+1]
                    pos=i+1
            C.append(pos)
        X["Cluster"]=C
        Centroids_new = X.groupby(["Cluster"]).mean()[[col2,col1]]
        if j == 0:
            diff=1
            j=j+1
        else:
            diff = (Centroids_new[col2] - Centroids[col2]).sum() + (Centroids_new[col1] - Centroids[col1]).sum()
            print(diff.sum())
        Centroids = X.groupby(["Cluster"]).mean()[[col2,col1]]

        for m in range(K):
            k_sum = X[m+1].sum()
        # for k in range(K):
        #     color = np.random.rand(1,3)
        #     data=X[X["Cluster"]==k+1]
        #     plt.scatter(data[col1],data[col2],c=color)
        # plt.scatter(Centroids[col1],Centroids[col2],c='red')
        # plt.xlabel(col1)
        # plt.ylabel(col2)
        # plt.show()
    elbow.append(k_sum)
    return elbow

names_n = ['id_num', 'outcome', 'rad', 'texture', 'perim', 'area', 'smooth', 'compact', 'concave', 'concave_points',
            'sym', 'fractal_dim', \
            'rad_SE', 'texture_SE', 'perim_SE', 'area_SE', 'smooth_SE', 'compact_SE', 'concave_SE',
            'concave_points_SE', 'sym_SE', 'fractal_dim_SE', \
            'rad_worst', 'texture_worst', 'perim_worst', 'area_worst', 'smooth_worst', 'compact_worst',
            'concave_worst', 'concave_points_worst', 'sym_worst', 'fractal_dim_worst']

df = pd.read_csv('wdbc.data', index_col=False,header=None, names= names_n)
df.head()

X = df[["rad","compact"]]
#Visualise data points
# plt.scatter(X["rad"],X["compact"],c='black')
# plt.xlabel('rad')
# plt.ylabel('compact')
# plt.show()

#number of clusters
plot_elbow = []
for K in range(2,10,1):
    plot_elbow.append(clustering(X, 'rad', 'compact', K))

print(plot_elbow)