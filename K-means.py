
### Author : Piyush Sharma
# --> Implementation of **K-MEANS** algorithim.
"""

#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#Reading the dataset 
iris = pd.read_csv('https://raw.githubusercontent.com/Piyush9323/NaiveBayes_in_Python/main/iris.csv')
#iris.head()
iris.tail()

iris.describe()

# Preparing input with 4 features from dataset
X = iris.iloc[:,1:5].values
X.shape

# Preparing Expected output for comparison
Y = iris.iloc[:,-1].values
for i in range(150):
    if i < 50 :
        Y[i] = 0
    elif i < 100 :
        Y[i] = 1
    else :
        Y[i] = 2
#Y

#number of training datapoints
m = X.shape[0]  

#number of features
n = X.shape[1]  

# number of iterations
n_iterations = 100

# number of clusters
k = 3
print(m,n,k)

data = iris.iloc[:,1:3].values
plt.scatter(data[:,0], data[:,1], c = 'black', label = 'Unclustered Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# It claculates Eucledian distance between two points
def E_distance(X1,X2):
    d = sum((X1 - X2)**2)**0.5
    return d

# K-Means Algorithm
def K_Means(data):

    # choosing centroids randomly 
    random.seed(121)
    centroids = {}
    for i in range(k):
        rand = random.randint(0,m-1)
        centroids[i] = data[rand]
        
    # creating output dictionary
    for iteration in range(n_iterations):
        classes = {}
    
        # classes are initialising with classKey as indices.
        for class_key in range(k):
            classes[class_key] = []

        # finding the distance of each data point with 3 centroids assigned recently.  
        p = 0   
        for data_point in data:
            distance = []
            for centroid in centroids:
                temp_dis = E_distance(data_point, centroids[centroid])
                distance.append(temp_dis)
            
            # finding the centroid with minimum distance from the point and append it into that centroid class.
            min_dis = min(distance)
            min_dis_index = distance.index(min_dis)
            classes[min_dis_index].append(data_point)
            Y[p] = min_dis_index
            p += 1

        # new centroids are formed by taking the mean of the data in each class.
        for class_key in classes:
            class_data = classes[class_key]
            new_centroids = np.mean(class_data, axis = 0)
            centroids[class_key] = list(new_centroids)

    return classes, centroids

# Running K-Means algorithm
classes, centroids = K_Means(X)
classes

# plotting the clustered data
color = ['red','blue','green']
labels = ['Iris-setosa','Iris-versicolour','Iris-virginica']

for i in range(k):
    x = list(list(zip(*classes[i]))[0])
    y = list(list(zip(*classes[i]))[1])
    plt.scatter(x, y, c = color[i], label = labels[i])

# plotting centroids of clusters
cv = centroids.values()
plt.scatter(list(list(zip(*cv))[0]), list(list(zip(*cv))[1]), s = 100, c = 'cyan',label= 'centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

"""# **K-MEANS** algorithm using inbuilt-function."""

from sklearn.cluster import KMeans

k_means = KMeans(n_clusters = 3, max_iter = 100, random_state = 15)
predicted = k_means.fit_predict(X)
predicted

# plotting the clusters
plt.scatter(X[predicted == 0, 0], X[predicted == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[predicted == 1, 0], X[predicted == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[predicted == 2, 0], X[predicted == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

# plotting the centroids of the clusters
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:,1], s = 100, c = 'cyan', label = 'Centroids')

plt.legend()

"""# **Summary** : Both clusters using my implementation and using Kmeans function from sklearn library gives nearly same results."""