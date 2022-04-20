import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from utils.general_util import *


def get_reconstruction(model, X, device='cpu', mean=None, std=None):
    '''Get embedding and reconstruction for autoencoder (encoder+decoder)'''
    model.to(device)
    model.eval()

    print("Starting inference...")
    with torch.no_grad():
        X = X.to(device)
        X_encoded = model.encode(X)
        X_recon = model.decode(X_encoded).cpu().detach().numpy()

    # redo standardization
    if mean is not None and std is not None:
        # print("Standardized...")
        X_recon = X_recon * std.numpy() + mean.numpy()

    X_encoded = X_encoded.cpu().detach().numpy()

    # if not 3D data apply reshape
    if len(X_recon.shape) < 3:
        # print("Reshape...")
        X_recon = X_recon.reshape((len(X_recon), -1, 3))

    return X_encoded, X_recon


def get_reconstruction_vae(model, X, device='cpu', mean=None, std=None):
    '''Get embedding and reconstruction for VAE'''
    model.to(device)
    model.eval()

    print("Starting inference...")
    with torch.no_grad():
        X = X.to(device)
        X_recon, X_encoded = model(X)
        X_encoded = X_encoded.cpu().detach().numpy()
        X_recon = X_recon.cpu().detach().numpy()

    # redo standardization
    if mean is not None and std is not None: 
        # print("Standardized...")
        X_recon = X_recon * std.numpy() + mean.numpy()

    # if not 3D data apply reshape
    if len(X_recon.shape) < 3:
        # print("Reshape...")
        X_recon = X_recon.reshape((len(X_recon), -1, 3))

    return X_encoded, X_recon


def generate_samples(model, n_sample=5, plot_z=True, 
                     xl=None, xr=None, yl=None, yr=None,
                     mean=None, std=None):
    '''Generate new samples with random z'''
    with torch.no_grad():
        if xl is not None and xr is not None \
            and yl is not None and yr is not None:
            x_range = np.linspace(xl, xr, num=n_sample, endpoint=True)
            y_range = np.linspace(yl, yr, num=n_sample, endpoint=True)
            z = torch.tensor(list(zip(x_range, y_range))).float()
        else:
            z = torch.randn(n_sample, 2)

        x = model.decode(z).cpu().detach().numpy()

    # if not 3D data apply reshape
    if len(x.shape) < 3:
        x = x.reshape((n_sample, -1, 3))

    # plot sampled points from embeddings
    if plot_z:
        plt.gcf().set_size_inches(5, 5)
        plt.scatter(z[:, 0], z[:, 1], s=10)

    # redo standardization
    if mean is not None and std is not None:
        x = x * std.numpy() + mean.numpy()
    return x, z


def tsne_transform(X, tsne_dim=2, perplexity=100):
    '''Apply TSNE on data'''
    tsne = TSNE(tsne_dim, perplexity=perplexity)
    return tsne.fit_transform(X)


def _plot_embedding(embeddings, y, palette):
    '''Plot 2D embedding (first 2 dimensions)'''

    fig, ax = plt.subplots(1, figsize=(9, 9))
    df_vis = pd.DataFrame({'dim1': embeddings[:, 0], 'dim2': embeddings[:, 1],
                           'label': y})
    g=sns.scatterplot(x='dim1', y='dim2', hue='label', 
                      palette=palette, data=df_vis, ax=ax, s=12)
    ax.set_title("Embeddings")
    g.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), ncol=2)


def plot_embedding(subj_data, palette, use_TSNE=False):
    if use_TSNE:
        _plot_embedding(subj_data.X_encoded_tsne, 
                        map_list_with_dict(subj_data.y, subj_data.bundle_num),
                        palette)
    else:
        _plot_embedding(subj_data.X_encoded, 
                        map_list_with_dict(subj_data.y, subj_data.bundle_num),
                        palette)