
"""
K-means clustering (Question)
Name: Maddhujeet Chandra
"""
import pandas as pd
import numpy as np
import random
import copy

def myKMeans(dataset, k = 3, seed = 42):
    """Returns the centroids of k clusters for the input dataset.
    
    Parameters
    ----------
    dataset: a pandas DataFrame
    k: number of clusters
    seed: a random state to make the randomness deterministic
    
    Examples
    ----------
    myKMeans(df, 5, 123)
    myKMeans(df)
    
    Notes
    ----------
    The centroids are returned as a new pandas DataFrame with k rows.
    
    """
    global flag
    flag = True
    numIterations = 4
    oldcentroids = getInitialCentroids(dataset,k, seed=42)
    labels = getLabels(dataset, oldcentroids)
    newcentroids = computeCentroids(dataset,labels)
    for i in range(numIterations):
        stopClustering(oldcentroids,newcentroids,numIterations)
        if (flag==False):
            break
        oldcentroids = newcentroids
        labels = getLabels(dataset,oldcentroids)
        newcentroids = computeCentroids(dataset,labels)
    cent_df = pd.DataFrame(data=newcentroids,columns=['Alcohol', 
                         'MalicAcid', 
                         'Ash', 
                         'AlcalinityAsh', 
                         'Magnesium', 
                         'TotPhenols', 
                         'Flavanoids', 
                         'NonflavanoidPhenols', 
                         'Proanthocyanins', 
                         'ColorIntensity', 
                         'Hue', 
                         'OD280/OD315Diluted', 
                         'Proline'])
    cent_df.index.name = 'Label'
    return cent_df
    


def getNumFeatures(dataset):
    """Returns a dataset with only numerical columns in the original dataset"""
    return dataset.select_dtypes(include=['float64','int'])


def getInitialCentroids(dataset, k, seed=42):
    """Returns k randomly selected initial centroids of the dataset"""
    global df
    df = dataset.copy()
    df.drop(columns=['y_pred1'],inplace=True,errors='ignore')
    init_centroids=[]
    global label
    label=[]
    for i in range(k):
        label.append(i)
        centroids=[]
        for j in range(len(df.columns)):
            centroids.append(random.uniform(-10,10))
        init_centroids.append(centroids)
        centroids=np.array(init_centroids)
    return centroids

def getLabels(dataset,centroids):
    """Assigns labels (i.e. 0 to k-1) to individual instances in the dataset.
    Each instance is assigned to its nearest centroid.
    """
    df1=df.copy()
    k=centroids.shape[0]
    for i in range(k):
        df['distance_from_{}'.format(i)]=np.sqrt(((df1 - centroids[i,:]) ** 2).sum(axis=1))
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in range(k)]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    labels = df['closest']
    return labels


def computeCentroids(dataset, labels):
    """Returns the centroids of individual groups, defined by labels, in the dataset"""
    k=len(label)
    centroids = getInitialCentroids(dataset,k)
    old_centroids = copy.deepcopy(centroids)
    for i in label:
        centroids[i,:] = np.mean(df.drop(columns=['distance_from_0', 'distance_from_1', 'distance_from_2', 'closest'],axis=1,errors='ignore')[labels == i])
    return centroids

def stopClustering(oldCentroids, newCentroids, numIterations, maxNumIterations = 100, tol = 1e-4):
    """Returns a boolean value determining whether the k-means clustering converged.
    Two stopping criteria: 
    (1) The distance between the old and new centroids is within tolerance OR
    (2) The maximum number of iterations is reached 
    """
    if np.sum((newCentroids - oldCentroids)/oldCentroids* 100.0) > tol:
        flag= False
        return flag 

    if (numIterations >= maxNumIterations):
        flag = False
        return flag

        