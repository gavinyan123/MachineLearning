import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class KCluster:
    def __init__(self, file = None, name = None):
        self.data = None
        self.name = name
        if file: 
            self.data = pd.read_csv(file,index_col=False,header=None, names= name)

    def plot_raw_data(self, malig, benign):
        #Plot all the columns of the data against each other
        for i in range(1,len(self.name),1):
            x = self.name[i]
            for j in range(i+1,len(self.name),1):
                y = self.name[j]
                X_mal = malig[[x,y]]
                X_ben = benign[[x,y]]
                #Visualise data points
                plt.scatter(X_mal[x],X_mal[y],c='red')
                plt.scatter(X_ben[x],X_ben[y],c='black')
                #plt.imshow(X, cmap='hot', interpolation='nearest')
                plt.xlabel(x)
                plt.ylabel(f'{y}')
                plt.show()

    #Find initial centroids based on statistical averages and percentiles
    def center(self, X, Y:int):
        Z = self.data.shape[0]
        p_list = [x for x in range(0, Z, int(Z/(Y+1)))]
        return X.iloc[p_list[1:Y+1]]

    #implementation of KMeans
    def KMeans(self, col1, col2, K):
        Z = self.data.shape[0]
        
        x = col1
        y = col2
        X = self.data[[x,y]]
        
        #setting centroids with center method
        centroids = self.center(X, K)

        #plot centroids
        plt.scatter(X[x], X[y],c='black')
        plt.scatter(centroids[x], centroids[y], c='red')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

        #Assign points to the closest centroid
        #Recompute Centroids
        #Repeat

        diff = 1
        j = 0

        while(diff!=0):
            XD = X
            i=1
            # for each centroid
            for index1,row_c in centroids.iterrows():
                ED=[]
                #measure distance from centroid to every point
                for index2,row_d in XD.iterrows():
                    d1=(row_c[x]-row_d[x])**2
                    d2=(row_c[y]-row_d[y])**2
                    d=np.sqrt(d1+d2)
                    ED.append(d)

                X[i]=ED # store distance vector as part of original Matrix
                i=i+1

            C=[]
            #find nearest centroid for each point
            for index,row in X.iterrows():
                min_dist=row[1]
                pos=1
                for i in range(K):
                    if row[i+1] < min_dist:
                        min_dist = row[i+1]
                        pos=i+1
                C.append(pos)
            X["Cluster"]=C
            Centroids_new = X.groupby(["Cluster"]).mean()[[y,x]]
            if j == 0:
                diff=1
                j=j+1
            else:
                diff = (Centroids_new[y] - centroids[y]).sum() + (Centroids_new[x] - centroids[x]).sum()
                print(diff.sum())
            centroids = X.groupby(["Cluster"]).mean()[[y,x]]

        for k in range(K):
            color = np.random.rand(1,3)
            print(color)
            data=X[X["Cluster"]==k+1]
            plt.scatter(data[x],data[y],c=color)
        plt.scatter(centroids[x], centroids[y], c="red")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

if __name__ == "__main__":
    names = ['id_num', 'outcome', 'rad', 'texture', 'perim', 'area', 'smooth', 'compact', 'concave', 'concave_points',
            'sym', 'fractal_dim', \
            'rad_SE', 'texture_SE', 'perim_SE', 'area_SE', 'smooth_SE', 'compact_SE', 'concave_SE',
            'concave_points_SE', 'sym_SE', 'fractal_dim_SE', \
            'rad_worst', 'texture_worst', 'perim_worst', 'area_worst', 'smooth_worst', 'compact_worst',
            'concave_worst', 'concave_points_worst', 'sym_worst', 'fractal_dim_worst']
    
    file = 'wdbc.data'

    k = KCluster(file, names)
    k.data['outcome'] = k.data['outcome'].map(lambda diag: bool(diag == "M"))  # M being cancerous
    #choose a column to sort the data by, this makes it easier to pick initial centroil positions
    k.data.sort_values( by=['area'], inplace=True)

    for K in range(2,10,1):
        k.KMeans('rad', 'compact', K)

