# -----------------------------------------------------------------------------
# Distributed under the GNU General Public License.
#
# Contributors: Nicolas Melo (nmo@cin.ufpe.br)
#               Rodrigo Sousa (--)
#               Adenilton Silva ajsilva@cin.ufpe.br
# -----------------------------------------------------------------------------
# File description:
#
# Auxiliary codes
# -----------------------------------------------------------------------------

import math
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, datasets
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances

def plot_iris():
    """
    Apply preprocess to Iris dataset and plot
    """
    
    samples, classes = load_iris(classes=[0,1], features=[0,1])
    samples = standardization(samples)
    samples = normalization(samples)
    
    x = samples[:, :1]
    y = samples[:, 1:]
    x = np.reshape(x, (len(x),))
    y = np.reshape(y, (len(y),))
    
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, c=classes[:100], cmap='viridis', marker='^')
    
    #plt.savefig("test.svg", format="svg")
    
    plt.show()
    
def standardization(X):
    """
    Standardize a dataset along any axis.
    Center to the mean and component wise scale to unit variance
    """
    
    return preprocessing.scale(X)

def normalization(X, norm='l2'):
    """
    Scale input vectors individually to unit norm (vector length)
    """
    
    return preprocessing.normalize(X, norm)

def preprocess(X):
    """
    Perform standardization and normalization on a dataset
    """
    
    X = standardization(X)
    X = normalization(X)
    
    return X

def load_wine(classes=None, features=None):
    """
    Load 'classes' and 'features' of the wine dataset
    """
    
    wine = datasets.load_wine()
    
    if features == None:
        X = wine.data
    else:
        X = wine.data[:, features]
        
    y = wine.target
    
    X = {i : X[i] for i in range(len(X))}
    
    if classes != None:
        for i in range(len(y)):
            if y[i] not in classes:
                del X[i]
                
    X = [v for v in X.values()]
    
    return X, y

def load_iris(classes=None, features=None):
    """
    Load 'classes' and 'features' of the iris dataset
    """
    
    iris = datasets.load_iris()
    
    if features == None:
        X = iris.data
    else:
        X = iris.data[:, features]
    
    y = iris.target
    
    X = {i : X[i] for i in range(len(X))}
    
    if classes != None:
        for i in range(len(y)):
            if y[i] not in classes:
                del X[i]
    
    X = [v for v in X.values()]
    
    return X, y

def load_sample(sample, preproc=True):
    """
    Load individual sample from iris dataset
    """
    
    classes = [0,1]
    features = [0,1]
    X, y = load_iris(classes=classes, features=features)
    
    if preproc:
        X = preprocess(X)
    
    return X[sample], y[sample]

def get_theta(sample):
    """
    Returns the angle associated with the sample
    """
    
    if sample[0] < 0 and sample[1] < 0:
        value = sample[1]
        theta = math.acos(value)*2+math.pi
    elif sample[1] < 0:
        value = sample[1]
        theta = math.asin(value)*2
    else:
        value = sample[0]
        theta = math.acos(value)*2
    
    return theta

def centroid(X):
    """
    Gets the centroid os a dataset with 2 features
    """
    
    n = len(X)
    xs = X[:,:1]
    ys = X[:,1:]
    
    return [(sum(xs)/n)[0], (sum(ys)/n)[0]]

def centroid2(X):
    """
    Gets the centroid os a dataset with 4 features
    """
    
    n = len(X)
    x1 = X[:,:1]
    x2 = X[:,1:2]
    x3 = X[:,2:3]
    x4 = X[:,3:4]
    
    return [(sum(x1)/n)[0], (sum(x2)/n)[0], (sum(x3)/n)[0], (sum(x4)/n)[0]]

def closest(X, c):
    """
    Compute minimum distances between one point and a set of points.
    This function computes for each row in X, the index of the row of Y which 
    is closest (according to the specified distance). The minimal distances are 
    also returned.
    """
    
    closest, _ = pairwise_distances_argmin_min([c], X, metric='l2')

    return closest[0]

def minDistances(X, c, n):
    """
    Compute the distance matrix from a vector array X and optional Y.
    This method takes either a vector array or a distance matrix, and returns a 
    distance matrix. If the input is a vector array, the distances are computed. 
    If the input is a distances matrix, it is returned instead.
    """
    
    dists = pairwise_distances([c], X, metric='l2')
    
    return np.argsort(dists)[0][:n]

def saveFig(path, fig):
    """
    Saves 'fig' to the specified path
    """
    
    n = 1
    while os.path.exists(path + "%s" %n):
        n += 1
        
    filename = path + str(n)
    fig.savefig(filename, format="svg")
    
def saveCsvFile(filename, dict, index):
    """
    Save dict in csv
    """
    
    df = pd.DataFrame(dict)
    n = 1
    while os.path.exists(filename + "%s.csv" %n):
        n += 1
        
    filename = filename + str(n) + "_" + str(tuple(index)) + ".csv"
    df.to_csv(filename, index=False)
    
def readCsvFile(filename):
    """
    Read a csv file from a 'filename'
    """
    
    with open(filename, 'r') as csvFile:
        df = pd.read_csv(csvFile)
    
    return df