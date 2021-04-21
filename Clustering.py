import numpy as np
from sklearn.manifold import TSNE


def tsne_cluster(X=None, Y=None, components=2, visualize=True):
    
    tsne = TSNE(n_components=2)
    x = tsne.fit_transform(X)
    return x
