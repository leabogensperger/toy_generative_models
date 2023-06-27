from sklearn.datasets import make_moons, make_circles
import numpy as np
import torch

def generate(n_samples, dataset):
    """ Generate a 2d toy dataset (8 gaussians, bunny, moons) of n_samples.
    """
    z = torch.randn(n_samples, 2)

    if dataset == 'moons':
        x = make_moons(n_samples, noise=0.05, shuffle=True)[0]
        x = torch.from_numpy(x)

    elif dataset == '8gaussians':
        scale = 4
        sq2 = 1 / np.sqrt(2)
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (sq2, sq2), (-sq2, sq2), (sq2, -sq2), (-sq2, -sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x, y in centers])
        
        x = sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == 'bunny':
        a1 = 0.25
        a2 = 0.25
        a3 = 0.5

        mu1 = np.array([[2.5],[1]])
        mu2 = np.array([[1],[0]])
        mu3 = np.array([[-.5],[0.5]])

        sig1 = np.array([[1,0.15],[0.15,.05]])
        sig2 = np.array([[1,-0.15],[-0.15,.05]])
        sig3 = np.array([[1,0],[0,0.25]])

        L1_sig = np.linalg.cholesky(sig1)
        L2_sig = np.linalg.cholesky(sig2)
        L3_sig = np.linalg.cholesky(sig3)

        x1 = mu1 + L1_sig@np.random.randn(2,int(a1*n_samples))
        x2 = mu2 + L2_sig@np.random.randn(2,int(a2*n_samples))
        x3 = mu3 + L3_sig@np.random.randn(2,int(a3*n_samples))

        x = torch.from_numpy(np.hstack([x1,x2,x3]).T)

    else:
        raise NotImplementedError('Dataset type <%s> is not implemented!' %dataset)

    return x.float()