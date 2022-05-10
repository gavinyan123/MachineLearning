#import libraries
from colorsys import yiq_to_rgb
import copy
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.datasets import make_blobs

#majority clustering
#if false>true then everything is false
#if true>false then everything is true
def majority_cluster(df, K):
    cluster_data = df
    for i in range(1,K):
        # print(i)
        cluster_point = cluster_data.loc[cluster_data['Cluster'] == i]
        values = cluster_point['outcome'].value_counts().keys().tolist()
        counts = cluster_point['outcome'].value_counts().tolist()
        # print(values, counts, "printing here")
        # if(i == 4):
        #     cluster_point['outcome'] = True
        #     cluster_data.loc[cluster_data['Cluster'] == i] = cluster_point
        #     # print(cluster_point)
        #     continue
        '''
        check first index of values
        first index is always the larger number
        check for length of values to make sure there is a True and False
            if length is 1 then append either true or false
        check for length of counts 
            if length is 1 then append 0
        '''
        if(values[0] == True):
            if(len(values) == 1):
                values.append(False)
            else:
                values[1] = False
        else:
            values[0] = False
            if(len(values) == 1):
                values.append(True)

        if(len(counts) == 1):
            counts.append(0)

        '''
        creating a data frame to compare values and their counts
        separate the data frame into either True or False
        '''
        data = {values[0]:[counts[0]], values[1]:[counts[1]]}
        cluster_df = pd.DataFrame(data)
        # print(cluster_df)

        truth = cluster_df.loc[:,True]
        notTruth = cluster_df.loc[:,False]
        # print(truth, notTruth)

        '''
        if false>true then everything is false
        if true>false then everything is true
        '''
        if(notTruth.iloc[0] > truth.iloc[0]):
            cluster_point['outcome'] = False
            cluster_data.loc[cluster_data['Cluster'] == i] = cluster_point
            # print(cluster_point)
        else:
            cluster_point['outcome'] = True
            cluster_data.loc[cluster_data['Cluster'] == i] = cluster_point
            # print(cluster_point)
    return cluster_data

def stratify(data, target, n):
    array = data.values
    y = data[target].values
    
    unique, counts = np.unique(data[target].values, return_counts=True)
    new_counts = counts * (n/sum(counts))
    new_counts = fit_new_counts_to_n(new_counts, n)
    
    selected_count = np.zeros(len(unique))
    selected_row_indices = []
    for i in range(array.shape[0]):
        if sum(selected_count) == sum(new_counts):
            break
        cr_target_value = y[i]
        cr_target_index = np.where(unique==cr_target_value)[0][0]
        if selected_count[cr_target_index] < new_counts[cr_target_index]:
            selected_row_indices.append(i)
            selected_count[cr_target_index] += 1
    row_indices_mask = np.array([x in selected_row_indices for x in np.arange(array.shape[0])])
    
    return pd.DataFrame(array[row_indices_mask], columns=data.columns)

def fit_new_counts_to_n(new_counts, n):
    decimals = [math.modf(x)[0] for x in new_counts]
    integers = [int(math.modf(x)[1]) for x in new_counts]
    arg_max = np.array(map(np.argmax, decimals))
    sorting_indices =  np.argsort(decimals)[::-1][:n]
    for i in sorting_indices:
        if sum(integers) < n:
            integers[i] += 1
        else:
            break
    return integers

#all points of data subtracted by min divided by max minus min
def Normalize(data):
    return (data - data.min())/(data.max()-data.min())

class LinearRegress:
    def __init__(self, data, col1, col2, lr, iter):
        self.data = data
        self.col1 = col1
        self.col2 = col2
        self.lr = lr
        self.iter = iter
    def linear_regression(self):
        N = len(self.data)
        x = self.data[self.col1]
        y = self.data[self.col2]
        x_mean = x.mean()
        y_mean = y.mean()
        
        B1_num = ((x - x_mean) * (y - y_mean)).sum()
        B1_den = ((x - x_mean)**2).sum()
        B1 = B1_num / B1_den
        
        B0 = y_mean - (B1*x_mean)
        
        reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))
        return (B0, B1, reg_line)

    def gradient_descent(self):
        points = np.array(self.data[[self.col1, self.col2]])
        learning_rate = self.lr
        b = 0
        m = 0
        num_iterations = self.iter
        D_b = 0
        D_m = 0
        #iterate num_iterations times
        for i in range(num_iterations):
            N = float(len(points))
            #iterate number of points times
            for j in range(0, len(points)):
                x = points[j, 0]
                y = points[j, 1]
                #partial derivative w.r.t. b
                #sum of Y - Ypred times (-2/N)
                #(-2/n)E(Y-Ypred)
                D_b += (-2/N) * (y-((m * x)+ b))
                #partial derivative w.r.t. m
                #sum of X times Y - Ypred times (-2/N)
                #(-2/n)EX(Y-Ypred)
                D_m += (-2/N) * x * (y-((m * x)+ b))
                #update b and m
                b = b - (learning_rate * D_b)
                m = m - (learning_rate * D_m)
        return b,m

    def calculate_error(b, m, points):
        error = 0
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]
            error = error + (y - (m * x + b))**2
        return error/float(len(points))

class kcluster:
    def __init__(self, data, col1, col2):
        self.data = data
        self.col1 = col1
        self.col2 = col2
    # Select random observation as centroids
    def get_init_centroids(self, K):
        N = self.data.shape[0]
        percentile_list  = [x for x in range(0,N,int(N/(K+1)))]    
        Centroids = self.data.iloc[percentile_list[1:K+1]]
        # Centroids = (data.sample(n=K))
        # plt.scatter(data["rad"],X["compact"],c='black')
        # plt.scatter(Centroids["rad"],Centroids["compact"],c='red')
        # plt.xlabel('rad')
        # plt.ylabel('compact')
        # plt.show()
        return Centroids

    def plot_init_centroids(self, K):
        Centroids = self.get_init_centroids(K)
        plt.scatter(self.data[self.col1],self.data[self.col2],c='black')
        plt.scatter(Centroids[self.col1],Centroids[self.col2],c='red')
        plt.xlabel(self.col1)
        plt.ylabel(self.col2)
        plt.show()

    def clustering(self, K):
        diff = 1
        j=0
        Centroids = self.get_init_centroids(K)
        X = self.data
        elbow_plot = []
        
        while(diff!=0):
            XD=X
            i=1
            for index1,row_c in Centroids.iterrows():
                ED=[]
                for index2,row_d in XD.iterrows():
                    d1=(row_c[self.col1]-row_d[self.col1])**2
                    d2=(row_c[self.col2]-row_d[self.col2])**2
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
            cluster_d = X.loc[X['Cluster'] == 2]
            # print(cluster_d['outcome'].value_counts())

            #find inertia/elbow method
            z=0
            elbow = []
            elbow_sum = []

            for index3, row_e in Centroids.iterrows():
                centroid_point = Centroids.iloc[0]
                xd = X.loc[X['Cluster'] == z+1]
                for index4, row_f in xd.iterrows():
                    sum1 = (centroid_point[self.col1] - row_f[self.col1])**2
                    sum2 = (centroid_point[self.col2] - row_f[self.col2])**2
                    total = np.sqrt(sum1+sum2)
                    elbow.append(total)
                elbow_sum.append(sum(elbow))
                z+1
            elbow_plot.append(elbow_sum[0])

            Centroids_new = X.groupby(["Cluster"]).mean()[[self.col2,self.col1]]
            if j == 0:
                diff=1
                j=j+1
            else:
                diff = (Centroids_new[self.col2] - Centroids[self.col2]).sum() + (Centroids_new[self.col1] - Centroids[self.col1]).sum()
                # print(diff.sum())
            Centroids = X.groupby(["Cluster"]).mean()[[self.col2,self.col1]]

            # for k in range(K):
            #     color = np.random.rand(1,3)
            #     data=X[X["Cluster"]==k+1]
            #     plt.scatter(data[col1],data[col2],c=color)
            # plt.scatter(Centroids[col1],Centroids[col2],c='red')
            # plt.xlabel(col1)
            # plt.ylabel(col2)
            # plt.show()

        # elbow_points = elbow_plot[:8]
        # plt.plot(range(len(elbow_points)), elbow_points,'go--', linewidth=1.5, markersize=4)
        # plt.xlabel("Iterations")
        # plt.ylabel("Sum Squares")
        # plt.show()
        return elbow_plot, X

class calc_accuracy:
    def accuracy(orig_df, new_df):
        orig_df['accuracy'] = np.where(orig_df['outcome'] == new_df['outcome'], True, False)
        positives = orig_df.loc[orig_df['outcome'] == True]
        negatives = orig_df.loc[orig_df['outcome'] == False]
        # positives['true_positive'] = np.where(positives['outcome'] == positives['accuracy'], True, False)
        # negatives['true_negative'] = np.where(negatives['outcome'] == negatives['accuracy'], True, False)
        
        pos_val = positives['accuracy'].value_counts().keys().tolist()
        pos_count = positives['accuracy'].value_counts().tolist()
        
        if(pos_val[0] == True):
            if(len(pos_val) == 1):
                pos_val.append(False)
            else:
                pos_val[1] = False
        else:
            pos_val[0] = False
            if(len(pos_val) == 1):
                pos_val.append(True)

        if(len(pos_count) == 1):
            pos_count.append(0)
        
        pos_data = {pos_val[0]:[pos_count[0]], pos_val[1]:[pos_count[1]]}
        
        pos_df = pd.DataFrame(pos_data)
        pos_df['sum'] = pos_df.sum(axis=1)

        neg_val = negatives['accuracy'].value_counts().keys().tolist()
        neg_count = negatives['accuracy'].value_counts().tolist()

        if(neg_val[0] == True):
            if(len(neg_val) == 1):
                neg_val.append(False)
            else:
                neg_val[1] = False
        else:
            neg_val[0] = False
            if(len(neg_val) == 1):
                neg_val.append(True)

        if(len(neg_count) == 1):
            neg_count.append(0)

        neg_data = {neg_val[0]:[neg_count[0]], neg_val[1]:[neg_count[1]]}

        neg_df = pd.DataFrame(neg_data)
        neg_df['sum'] = neg_df.sum(axis=1)

        avg_pos = pos_df[True]/pos_df['sum']
        avg_neg = neg_df[True]/neg_df['sum']
        uar=(avg_neg + avg_pos)/2
        print(uar)
        return uar

    def reg_avg(orig_df, new_df):
        orig_df['accuracy'] = np.where(orig_df['outcome'] == new_df['outcome'], True, False)
        val = orig_df['accuracy'].value_counts().keys().tolist()
        count = orig_df['accuracy'].value_counts().tolist()
        
        if(val[0] == True):
            if(len(val) == 1):
                val.append(False)
            else:
                val[1] = False
        else:
            val[0] = False
            if(len(val) == 1):
                val.append(True)

        if(len(count) == 1):
            count.append(0)
        
        data = {val[0]:[count[0]], val[1]:[count[1]]}
        
        df = pd.DataFrame(data)
        df['sum'] = df.sum(axis=1)
        average = df[True]/df['sum']
        return average
'''
Z = w*x + b
a = sig(z) = 1 / (1+e^-z)
x1 = delta(Z)/delta(w)
'''

class LogisticRegress:
    def __init__(self, d_x, lr = 0.001, iters = 1000):
        self.lr = lr
        self.iters = iters
        self.m, self.n = d_x.shape
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def train(self, d_x, y):
        self.weights = np.zeros((self.n, 1))
        self.bias = 0

        for it in range(self.iters+1):
            y_pred = self.sigmoid(np.dot(d_x, self.weights) + self.bias)

            cost = -1/self.m * np.sum(y*np.log(y_pred) + (1-y)*np.log(1 - y_pred))

            dw = 1/self.m * np.dot(d_x.T, (y_pred - y))
            db = 1/self.m * np.sum(y_pred - y )

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self.weights, self.bias
    # def fit(self, d_x, y):
    #     pass

    def predict(self, d_x):
        y_pred = self.sigmoid(np.dot(d_x, self.weights) + self.bias)
        y_pred_lab = y_pred > 0.5

        return y_pred_lab

names_n = ['id_num', 'outcome', 'rad', 'texture', 'perim', 'area', 'smooth', 'compact', 'concave', 'concave_points',
            'sym', 'fractal_dim', \
            'rad_SE', 'texture_SE', 'perim_SE', 'area_SE', 'smooth_SE', 'compact_SE', 'concave_SE',
            'concave_points_SE', 'sym_SE', 'fractal_dim_SE', \
            'rad_worst', 'texture_worst', 'perim_worst', 'area_worst', 'smooth_worst', 'compact_worst',
            'concave_worst', 'concave_points_worst', 'sym_worst', 'fractal_dim_worst']

# sigmoid function = 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    df = pd.read_csv('wdbc.data', index_col=False,header=None, names= names_n)
    df['outcome'] = df['outcome'].map(lambda diag: bool(diag == "M"))  # M being cancerous

    

    #choose a column to sort the data by, this makes it easier to pick initial centroil positions
    df.sort_values( by=['area'], inplace=True)

    for location in range(len(df.columns)):
        if df.iloc[:,location].name == 'outcome':
            continue
        df.iloc[:,location] = Normalize(df.iloc[:,location])
    
    X=copy.deepcopy(df)

    for h in range(len(df)):
            if df['outcome'].iloc[h] == True:
                df['outcome'].iloc[h] = 1
            else:
                df['outcome'].iloc[h] = 0
    K=7

    new_X=copy.deepcopy(df)
    sns.regplot(x=X['rad'], y=X['outcome'], data=X, logistic=True, ci=None)
    plt.show()

    lin = LinearRegress(new_X, 'rad', 'smooth', 0.0001, 1000)
    logRegLine = LinearRegress(new_X, 'rad', 'outcome', 0.0001, 1000)
    kmeans = kcluster(X, 'rad', 'smooth')
   

    logData = new_X[['rad', 'smooth']].to_numpy().astype(float)
    y = new_X['outcome'].to_numpy().astype(float)
    y = y[:,np.newaxis]
    logReg = LogisticRegress(logData)
    w, b = logReg.train(logData, y)
    sig = []
    for log in range(len(logData)):
        z = (w*logData[log]) + b
        sig.append(logReg.sigmoid(z))
    plt.plot(sig)
    plt.show()


    B0, B1, reg_line = lin.linear_regression()
    print(reg_line)
    plt.scatter(X['rad'], X['smooth'])
    plt.plot(X['rad'], B0 + B1*X['rad'],c = 'r', linewidth=1.5, markersize=4)
    plt.xlabel('rad')
    plt.ylabel('smooth')
    plt.show()

    b,m = lin.gradient_descent()
    print(b,m)
    plt.scatter(X['rad'], X['smooth'])
    plt.plot(X['rad'], b + m*X['rad'],c = 'r', linewidth=1.5, markersize=4)
    plt.xlabel('rad')
    plt.ylabel('smooth')
    plt.show()

    b2, m2 = logRegLine.gradient_descent()
    z = (m2*new_X['rad']) + b2
    sig = 1/(1+np.exp(-z))
    # plt.scatter(new_X['rad'], new_X['outcome'])
    plt.plot(new_X['rad'], sig)
    plt.show()

    og_strat = stratify(X, 'outcome', 150)

    # kernelization(og_strat, 'rad', 'smooth')
    # plt.scatter(og_strat['rad'], og_strat['smooth'])
    # plt.plot(og_strat['rad'], kernel,'go--',c = 'r', linewidth=1.5, markersize=4)
    # plt.show()

    accuracy_data = []
    average_data = []

    for repeat in range(2,10):
        elbow, cluster = kmeans.clustering(repeat)
        use_cluster = copy.deepcopy(cluster)
        elbow_points = elbow[2:10]
        # plt.plot(range(len(elbow_points)), elbow_points,'go--', linewidth=1.5, markersize=4)
        # plt.xlabel("Iterations")
        # plt.ylabel("Sum Squares")
        # plt.show()
        new_cluster = majority_cluster(use_cluster, repeat)
        nc_strat = stratify(new_cluster, 'outcome', 150)

        list_acc=calc_accuracy.accuracy(og_strat, nc_strat)
        list_avg = calc_accuracy.reg_avg(og_strat, nc_strat)
        accuracy_data.append(list_acc)
        average_data.append(list_avg)

    plt.plot(range(2,10), accuracy_data,'go--', linewidth=1.5, markersize=4)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(range(2,10), average_data,'go--', linewidth=1.5, markersize=4)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()

    #Visualise data points
    plt.scatter(X["rad"],X["compact"],c='black')
    plt.xlabel('rad')
    plt.ylabel('compact')
    plt.show()

    #iterate through cluster K amount of times
    for k in range(2,K,1):
        kmeans.plot_init_centroids(k)
        kmeans.clustering(k)