#import libraries
import copy
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math

def linear_regression(data, col1, col2):
    N = len(data)
    x = data[col1]
    y = data[col2]
    x_mean = x.mean()
    y_mean = y.mean()
    
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
    
    B0 = y_mean - (B1*x_mean)
    
    reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))
    return (B0, B1, reg_line)

def gradient_descent(X,col1, col2):
    points = np.array(X[[col1, col2]])
    learning_rate = 0.005
    b = 0
    m = 0
    num_iterations = 1000
    for i in range(num_iterations):
        N = float(len(points))
        D_b = 0
        D_m = 0
        for j in range(0, len(points)):
            x = points[j, 0]
            y = points[j, 1]
            D_b += (-2/N) * (y-((m * x)+ b))
            D_m += (-2/N) * x * (y-((m * x)+ b))
            b = b - (learning_rate * D_b)
            m = m - (learning_rate * D_m)
    return b,m
    
# def gradient_descent(X, col1, col2):
#     points = X[[col1, col2]]
#     learning_rate = 0.005
#     initial_b = 0
#     initial_m = 0
#     num_iterations = 1000
#     print( "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, calculate_error(initial_b, initial_m, np.array(points))))
#     print ("Running...")
#     [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
#     print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, calculate_error(b, m, np.array(points))))
#     return [b,m]

# def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
#     b = initial_b
#     m = initial_m
#     for i in range(num_iterations):
#         b, m = step_gradient(b,m, np.array(points), learning_rate)
#     return [b,m]

# def step_gradient(current_b, current_m, points, learning_rate):
#     D_b = 0
#     D_m = 0
#     N = float(len(points))
#     for i in range(0, len(points)):
#         x = points[i, 0]
#         y = points[i, 1]
#         D_b += (-2/N) * (y-((current_m * x)+ current_b))
#         D_m += (-2/N) * x * (y-((current_m * x)+ current_b))
#         current_b = current_b - (learning_rate * D_b)
#         current_m = current_m - (learning_rate * D_m)
#     new_b = current_b
#     new_m = current_m
#     return [new_b, new_m]

def calculate_error(b, m, points):
    error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        error = error + (y - (m * x + b))**2
    return error/float(len(points))

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

def Normalize(data):
    return (data - data.min())/(data.max()-data.min())

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

def plot_init_centroids(data, K, col1, col2):
    Centroids = get_init_centroids(data, K)
    plt.scatter(data[col1],data[col2],c='black')
    plt.scatter(Centroids[col1],Centroids[col2],c='red')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()

def clustering(data, col1, col2, K):
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
        data = {values[0]:[counts[0]], values[1]:[counts[1]]}
        cluster_df = pd.DataFrame(data)
        # print(cluster_df)

        truth = cluster_df.loc[:,True]
        notTruth = cluster_df.loc[:,False]
        # print(truth, notTruth)

        if(notTruth.iloc[0] > truth.iloc[0]):
            cluster_point['outcome'] = False
            cluster_data.loc[cluster_data['Cluster'] == i] = cluster_point
            # print(cluster_point)
        else:
            cluster_point['outcome'] = True
            cluster_data.loc[cluster_data['Cluster'] == i] = cluster_point
            # print(cluster_point)
    return cluster_data

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

names_n = ['id_num', 'outcome', 'rad', 'texture', 'perim', 'area', 'smooth', 'compact', 'concave', 'concave_points',
            'sym', 'fractal_dim', \
            'rad_SE', 'texture_SE', 'perim_SE', 'area_SE', 'smooth_SE', 'compact_SE', 'concave_SE',
            'concave_points_SE', 'sym_SE', 'fractal_dim_SE', \
            'rad_worst', 'texture_worst', 'perim_worst', 'area_worst', 'smooth_worst', 'compact_worst',
            'concave_worst', 'concave_points_worst', 'sym_worst', 'fractal_dim_worst']

if __name__ == "__main__":
    df = pd.read_csv('wdbc.data', index_col=False,header=None, names= names_n)
    df['outcome'] = df['outcome'].map(lambda diag: bool(diag == "M"))  # M being cancerous

    #choose a column to sort the data by, this makes it easier to pick initial centroil positions
    df.sort_values( by=['area'], inplace=True)

    for location in range(len(df.columns)):
        if df.iloc[:,location].name == 'outcome':
            continue
        df.iloc[:,location] = Normalize(df.iloc[:,location])

    print(df)
    K=5

    X=copy.deepcopy(df)

    B0, B1, reg_line = linear_regression(X, 'rad', 'smooth')
    plt.scatter(X['rad'], X['smooth'])
    plt.plot(X['rad'], B0 + B1*X['rad'],'go--',c = 'r', linewidth=1.5, markersize=4)
    plt.xlabel('rad')
    plt.ylabel('concave')
    plt.show()

    b,m = gradient_descent(X, 'rad', 'smooth')
    print(b,m)
    plt.scatter(X['rad'], X['smooth'])
    plt.plot(X['rad'], b + m*X['rad'], 'go--',c = 'r', linewidth=1.5, markersize=4)
    plt.show()
    og_strat = stratify(X, 'outcome', 25)

    accuracy_data = []
    average_data = []

    for repeat in range(2,10):
        elbow, cluster = clustering(X, 'rad', 'compact', repeat)
        use_cluster = copy.deepcopy(cluster)
        elbow_points = elbow[2:8]
        # plt.plot(range(len(elbow_points)), elbow_points,'go--', linewidth=1.5, markersize=4)
        # plt.xlabel("Iterations")
        # plt.ylabel("Sum Squares")
        # plt.show()
        new_cluster = majority_cluster(use_cluster, repeat)
        nc_strat = stratify(new_cluster, 'outcome', 25)

        list_acc=accuracy(og_strat, nc_strat)
        list_avg = reg_avg(og_strat, nc_strat)
        accuracy_data.append(list_acc)
        average_data.append(list_avg)

    plt.plot(range(len(accuracy_data)), accuracy_data,'go--', linewidth=1.5, markersize=4)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(range(len(average_data)), average_data,'go--', linewidth=1.5, markersize=4)
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
        plot_init_centroids(X, k, 'rad', 'compact')
        clustering(X, 'rad', 'compact', k)