import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def tsne_cluster(X=None, Y=None, components=2, visualize=True, iterations=100):
    
    tsne = TSNE(n_components=components, n_iter=iterations)
    x = tsne.fit_transform(X, Y)
    x = x - np.min(np.min(x))
    x = x / np.max(np.max(x))
    if visualize:
        tsne_plot(x, Y)
        
    return x


def tsne_plot(X=None, Y=None):
    
    if X.shape[1] == 2:        
        tsne_2d_plot(X, Y)
    
    elif X.shape[1] == 3:    
        tsne_3d_plot(X, Y)
    
    else:            
        print("Cannot plot with " + str(X.shape[1]) + " dimension")
        
    return


def tsne_2d_plot(X=None, Y=None):
    
    fig, ax = plt.subplots(figsize=(30, 20))
    for i in range(X.shape[0]):
        
        ax.plot(X[i, 0], X[i, 1], color=[Y[i, 0], 0.4, Y[i, 1]], marker='o', linewidth=5)
    
    fig.show()
    return


def tsne_3d_plot(X=None, Y=None):
    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(projection='3d')
    for i in range(X.shape[0]):
        
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], color=[Y[i, 0], 0.4, Y[i, 1]], marker='o', linewidth=5)
    
    fig.show()
    return
