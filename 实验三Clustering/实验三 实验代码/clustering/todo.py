import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def kmeans(X, k):
    '''
    K-Means clustering algorithm

    Input:  x: data point features, N-by-P maxtirx
            k: the number of clusters

    OUTPUT:  idx: cluster label, N-by-1 vector
    '''
    def dis(vector1, vector2):
        return np.sqrt(sum((vector2 - vector1) ** 2))
    N, P = X.shape
    idx = np.zeros((N, 1))
    cent = np.zeros((k, P))
    for i in range(k):
        index = int(np.random.uniform(0, N))
        cent[i, :] = X[index, :]

    info = np.array(np.zeros((N, 2)))
    change_cent = True
    stop=0
    while change_cent:
        if(stop>100):
            break
        stop=stop+1
        print(stop)
        change_cent = False
        for i in range(N):
            mindis = 999999999.0
            minidx = 0
            for j in range(k):
                distance = dis(cent[j, :], X[i, :])
                if distance < mindis:
                    mindis = distance
                    info[i, 1] = mindis
                    minidx = j
                if info[i, 0] != minidx:
                    change_cent = True
                    info[i, 0] = minidx
            for j in range(k):
                indexes = np.nonzero(info[:, 0] == j)
                points = X[indexes]
                cent[j, :] = np.mean(points, axis=0)
    return info[:,0]

def spectral(W, k):
    '''
    Spectral clustering algorithm

    Input:  W: Adjacency matrix, N-by-N matrix
            k: number of clusters

    Output:  idx: data point cluster labels, N-by-1 vector
    '''
    N = W.shape[0]
    # for j in range(N):
    #     for i in range(N):
    #         if(W[j][i]!=0):
    #             print(j,i)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    sqrtD = np.power(np.linalg.matrix_power(D,-1),0.5)
    lap = np.dot(np.dot(sqrtD, L), sqrtD)
    lam, E = np.linalg.eig(lap)
    dim = len(lam)
    dictEigval = dict(zip(lam, range(0, dim)))
    kEig = np.sort(lam)[0:k]
    ix = [dictEigval[k] for k in kEig]
    X=E[:, ix]
    # ix=sorted(range(len(lam)), key=lambda x: lam[x])[:k]
    # X=[]
    # for i in range(k):
    #     X.append(E[ix[i]])
    # X=np.array(X)
    X=X.astype(float)
    idx = kmeans(X, k)
    return idx

def knn_graph(X, k, threshold):
    '''
    Construct W using KNN graph

    Input:  X:data point features, N-by-P maxtirx.
            k: number of nearest neighbour.
            threshold: distance threshold.

    Output:  W - adjacency matrix, N-by-N matrix.
    '''
    N = X.shape[0]
    W = np.zeros((N, N))
    aj = cdist(X, X, 'euclidean')
    for i in range(N):
        index = np.argsort(aj[i])[:(k+1)]  # aj[i,i] = 0
        W[i, index] = 1
    W[aj >= threshold] = 0
    return W
