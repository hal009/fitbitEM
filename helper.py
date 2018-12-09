import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
def plot_cluster(data, mu, sigma, gamma_z, epoch):
    ax = plt.gca()
    colors = []
    for x in range(len(data)):
        colors.append((gamma_z[0, x], 0, gamma_z[1, x]))
    ax.scatter(data[:,0], data[:,1], c = colors)
    ax.set_title(f"Epoch: {epoch}")
    K = mu.shape[0]
    
    for k in range(K): 
        lambda_, v = np.linalg.eig(sigma[k])
        lambda_ = np.sqrt(lambda_)
        cluster = Ellipse(xy=(mu[k, 0], mu[k, 1])
                             , width = lambda_[0]*2
                             , height= lambda_[1]*2
                             , angle = np.rad2deg(np.arccos(v[0, 0]))
                             , color = 'black')
        cluster.set_facecolor('none')
        ax.add_artist(cluster)
    
    plt.show()
def plot_cluster_3d(data, mu, sigma, gamma_z, epoch):
    K = mu.shape[0]
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = []
    for x in range(len(data)):
        colors.append((gamma_z[0, x], 0, gamma_z[1, x]))
    ax.scatter(data[:,0], data[:,1], data[:, 2], c = colors)
    for k in range(K):
        ax.plot([mu[k, 0]], [mu[k, 1]], [mu[k, 2]], c='yellow', markersize=2, marker='o')
    ax.set_title(f"Epoch: {epoch}")