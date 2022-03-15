#import libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# Select random observation as centroids
def get_init_centroids(data, K):
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

def plot_init_centroids(data, K):
    Centroids = get_init_centroids(data, K)
    plt.scatter(data["rad"],data["compact"],c='black')
    plt.scatter(Centroids["rad"],Centroids["compact"],c='red')
    plt.xlabel('rad')
    plt.ylabel('compact')
    plt.show()

def clustering(data, col1, col2, K, plot):
    diff = 1
    j=0
    Centroids = get_init_centroids(data, K)
    X = data
    elbow_plot = []
    
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

        #find inertia/elbow method
        z=0
        elbow = []
        elbow_sum = []

        for index3, row_e in Centroids.iterrows():
            centroid_point = Centroids.iloc[0]
            xd = X.loc[df['Cluster'] == z+1]
            for index4, row_f in xd.iterrows():
                sum1 = (centroid_point[col1] - row_f[col1])**2
                sum2 = (centroid_point[col2] - row_f[col2])**2
                total = np.sqrt(sum1+sum2)
                elbow.append(total)
            elbow_sum.append(sum(elbow))
            z+1
        elbow_plot.append(elbow_sum[0])

        Centroids_new = X.groupby(["Cluster"]).mean()[[col2,col1]]
        if j == 0:
            diff=1
            j=j+1
        else:
            diff = (Centroids_new[col2] - Centroids[col2]).sum() + (Centroids_new[col1] - Centroids[col1]).sum()
            # print(diff.sum())
        Centroids = X.groupby(["Cluster"]).mean()[[col2,col1]]

        if(plot ==1):
            for k in range(K):
                color = np.random.rand(1,3)
                data=X[X["Cluster"]==k+1]
                plt.scatter(data[col1],data[col2],c=color)
            plt.scatter(Centroids[col1],Centroids[col2],c='red')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.show()
    elbow_points = elbow_plot[:10]
    plt.plot(range(len(elbow_points)), elbow_points,'go--', linewidth=1.5, markersize=4)
    plt.xlabel("Num Clusters(K)")
    plt.ylabel("Sum Squares")
    plt.show()
names_n = ['id_num', 'outcome', 'rad', 'texture', 'perim', 'area', 'smooth', 'compact', 'concave', 'concave_points',
            'sym', 'fractal_dim', \
            'rad_SE', 'texture_SE', 'perim_SE', 'area_SE', 'smooth_SE', 'compact_SE', 'concave_SE',
            'concave_points_SE', 'sym_SE', 'fractal_dim_SE', \
            'rad_worst', 'texture_worst', 'perim_worst', 'area_worst', 'smooth_worst', 'compact_worst',
            'concave_worst', 'concave_points_worst', 'sym_worst', 'fractal_dim_worst']

df = pd.read_csv('wdbc.data', index_col=False,header=None, names= names_n)
df['outcome'] = df['outcome'].map(lambda diag: bool(diag == "M"))  # M being cancerous
#choose a column to sort the data by, this makes it easier to pick initial centroil positions
df.sort_values( by=['area'], inplace=True)

K=10

X=df

#Visualise data points
plt.scatter(X["rad"],X["compact"],c='black')
plt.xlabel('rad')
plt.ylabel('compact')
plt.show()

#iterate through cluster K amount of times
for k in range(2,K,1):
    plot_init_centroids(X, k)
    clustering(X, 'rad', 'compact', k,1)