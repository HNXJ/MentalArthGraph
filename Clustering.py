import os
import numpy as np
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def pca_cluster(X=None, Y=None, components=2, visualize=True, tit=None, save=False, name=""):
    
    pca = PCA(n_components=components)
    x = pca.fit_transform(X)
    x = x - np.min(np.min(x))
    x = x / np.max(np.max(x))
    
    if visualize:
        cluster_plot(x, Y, tit, save, name)
        
    return x


def tsne_cluster(X=None, Y=None, components=2, visualize=True, iterations=100, tit=None, save=False, name=""):
    
    tsne = TSNE(n_components=components, n_iter=iterations, perplexity=7,
                learning_rate=100, init='random')
    x = tsne.fit_transform(X, Y)
    x = x - np.min(np.min(x))
    x = x / np.max(np.max(x))
    
    if visualize:
        cluster_plot(x, Y, tit, save, name)
        
    return x


def cluster_plot(X=None, Y=None, tit=None, save=False, name=""):
    
    if X.shape[1] == 2:        
        scatter_2d_plot(X, Y, tit, save, name)
    
    elif X.shape[1] == 3:    
        scatter_3d_plot(X, Y, tit, save, name)
    
    else:            
        print("Cannot plot with " + str(X.shape[1]) + " dimension")
        
    return


def scatter_2d_plot(X=None, Y=None, tit=None, save=False, name=""):
    
    fig, ax = plt.subplots(figsize=(30, 20))
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], color=[Y[i, 0], 0.4, Y[i, 1]], marker='o', linewidth=15 - Y[i, 1]*7)
        if Y[i, 1] == 1:
            ax.text(X[i, 0], X[i, 1] + X[i, 1]*0.02, str(int(i/(X.shape[0]/20))))
        else:
            ax.text(X[i, 0], X[i, 1] - X[i, 1]*0.02, str(int(i/(X.shape[0]/20))))
        
    ax.grid(True)
    ax.set_title(tit)
    fig.show()
    if save:
        try:
            fig.savefig("Plots/" + name + ".png")
        except:
            os.mkdir("Plots")
            fig.savefig("Plots/" + name + ".png")
    return


def scatter_3d_plot(X=None, Y=None, tit=None, save=False, name=""):
    
    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(projection='3d')
    for i in range(X.shape[0]):
        
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], color=[Y[i, 0], 0.4, Y[i, 1]], marker='o', linewidth=12 - Y[i, 1]*6)
    
    ax.set_title(tit)
    fig.show()
    if save:
        try:
            fig.savefig("Plots/" + name + ".png")
        except:
            os.mkdir("Plots")
            fig.savefig("Plots/" + name + ".png")
    return

