import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.optim import Adam

import base_model

class InvertibleLayer(nn.Module):
    def __init__(self, size_z, size_hidden, mask):
        super(InvertibleLayer, self).__init__()
        
        self.size_z = size_z
        self.size_hidden = size_hidden

        self.s = nn.Sequential(
            nn.Linear(self.size_z, self.size_hidden),
            nn.ELU(), 
            nn.Linear(self.size_hidden, self.size_hidden),
            nn.ELU(),
            nn.Linear(self.size_hidden, self.size_z),
        )
        
        self.t = nn.Sequential(
            nn.Linear(self.size_z, self.size_hidden),
            nn.ELU(), 
            nn.Linear(self.size_hidden, self.size_hidden),
            nn.ELU(),
            nn.Linear(self.size_hidden, self.size_z),
        )

        self.mask = mask[None,:]

    def forward(self, z):
        z_mask = z*self.mask
        s = self.s(z_mask)
        t = self.t(z_mask)
        x = z_mask + (1 - self.mask) * (z*torch.exp(s) + t)

        log_det_jac = ((1 - self.mask) * s).sum(-1)
        return x, log_det_jac
       

    def inverse(self, x):
        x_mask = x*self.mask
        s = self.s(x_mask)
        t = self.t(x_mask)
        z = x_mask + (1 - self.mask) * (x-t)*torch.exp(-s)

        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)
        return z, inv_log_det_jac

    def check_invertibility(self, n_samples):
        z = torch.randn(n_samples,self.size_z)
        print('Invertibility of layer s.t. z==T^{-1}(T(z)):', torch.allclose(z,self.inverse(self.forward(z)[0])[0]))
     

class NormalizingFlow(base_model.BaseModel):
    def __init__(self, n_layers, size_z, size_hidden, masks, prior):
        super(NormalizingFlow, self).__init__()
        self.device = masks.device 
        
        self.layers = nn.ModuleList()
        for layer in range(n_layers):
            self.layers.append(InvertibleLayer(size_z=size_z, size_hidden=size_hidden, mask=masks[layer]))
        self.prior_mu, self.prior_sigma = prior[0], prior[1] 

    def log_multivariate(self, tmp):
        return -0.5*(tmp**2).sum(-1)

    def backward(self, x):
        # sample x -> z, eval p(z)
        log_det = torch.zeros((x.shape[0])).to(x.device)
        for layer in self.layers:
            x, log_det_layer = layer.inverse(x)
            log_det += log_det_layer
        return x, log_det

    def forward(self, z):
        # sample z -> x, eval p(x)
        log_det = torch.zeros((z.shape[0])).to(z.device)
        for layer in reversed(self.layers):
            z, log_det_layer = layer(z)
            log_det += log_det_layer
            
        return z, log_det

    def log_pz_2DGauss(self, z):
        # evaluate log-pdf of prior distribution
        t1 = -np.log(2*np.pi) - 0.5*torch.linalg.slogdet(self.prior_sigma).logabsdet # constant terms 
        z_mu = z[:,:,None]-self.prior_mu[None,:,:]
        t2 = -0.5*torch.einsum('bjk,bj->bk',z_mu,torch.einsum('ij,bjk->bi',torch.linalg.inv(self.prior_sigma + 1e-6), z_mu))
        return t1 + t2

    def fit(self, x, max_iter, lr):
        optim = Adam(params=self.parameters(), lr=lr)
        self.train()
        
        loss_total, loss_pz, loss_log_det = [], [], []
        for iter_ in range(max_iter):
            z, sum_log_det = self.backward(x)
            
            log_pz = self.log_pz_2DGauss(z) # simple 2D gaussian for prior distribution
            loss = -log_pz.mean() - sum_log_det.mean() 

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_total.append(loss.item())
            loss_pz.append(-log_pz.mean().item())
            loss_log_det.append(-sum_log_det.mean().item())
    
            x = shuffle(x)
        
        return [loss_total, loss_pz, loss_log_det]
    
    def sample(self, n_samples):
        self.eval()
        L = torch.linalg.cholesky(self.prior_sigma)
        z_samples = self.prior_mu[None,:,0] + torch.einsum('ij,bj->bi', L, torch.randn(n_samples,2).to(self.device))
        x_samples, samples_log_det = self.forward(z_samples)
        return x_samples

def estimate_density(x, cfg, test_invertible_layer=False):
    if test_invertible_layer:
        masks = torch.nn.functional.one_hot(torch.tensor([i % 2 for i in range(4)])).float()
        T1 = InvertibleLayer(size_z=2, size_hidden=16, mask=masks[0])
        T1.check_invertibility(n_samples=3)

    masks = torch.nn.functional.one_hot(torch.tensor([i % 2 for i in range(cfg.n_layers)])).float().to(x.device)

    # prior distribution to which flow should map
    prior = [torch.zeros((2,1),device=x.device), torch.eye(2).to(x.device)]

    normflow = NormalizingFlow(n_layers=cfg.n_layers, size_z=cfg.size_z, size_hidden=cfg.size_hidden, masks=masks, prior=prior)
    normflow = normflow.to(x.device)
    print('Normalizing Flow: #learnable params=%d' %(normflow.count_parameters()))

    losses = normflow.fit(x,max_iter=cfg.max_iter,lr=cfg.lr)

    # sample from flow
    samples = normflow.sample(n_samples=cfg.n_samples)


    # plotting
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    gs = fig.add_gridspec(3,cfg.n_layers+1)

    # total loss and p_Z vs. log|\nabla T^-1(x)|
    f_ax11 = fig.add_subplot(gs[0, 0])
    f_ax11.set_title('Training loss'), f_ax11.set_xlabel('iterations')
    f_ax11.plot(np.asarray(losses[0]))

    f_ax12 = fig.add_subplot(gs[0, 1])
    f_ax12.set_title('$p_Z$($T^{-1}$(x))'), f_ax12.set_xlabel('iterations')
    f_ax12.plot(np.asarray(losses[1]))

    f_ax13 = fig.add_subplot(gs[0, 2])
    f_ax13.set_title(r'log det $\nabla$ $T^{-1}$(x)'), f_ax13.set_xlabel('iterations')
    f_ax13.plot(np.asarray(losses[1]))

    # visualize output of individual layers
    f_ax21 = fig.add_subplot(gs[1, 0])
    f_ax21.scatter(x[:,0].detach().cpu().numpy(), x[:,1].detach().cpu().numpy(),marker='.'), f_ax21.set_title('Original Data Samples')

    x_trans = x.clone()
    for layer_idx, layer in enumerate(normflow.layers):
        x_trans, _ = layer.inverse(x_trans)

        f_ax2layer = fig.add_subplot(gs[1,1+layer_idx])
        f_ax2layer.scatter(x_trans[:,0].detach().cpu().numpy(), x_trans[:,1].detach().cpu().numpy(),marker='.')
        f_ax2layer.set_title('Layer %i' %(layer_idx+1))
    
    # plot samples
    f_ax31 = fig.add_subplot(gs[2, 0])
    f_ax31.scatter(x[:,0].detach().cpu().numpy(), x[:,1].detach().cpu().numpy(),marker='.'), f_ax31.set_title('Original Data Samples')
    f_ax32 = fig.add_subplot(gs[2, 1], sharex=f_ax31, sharey=f_ax31)
    f_ax32.scatter(samples[:,0].detach().cpu().numpy(), samples[:,1].detach().cpu().numpy(),marker='.'), f_ax32.set_title('Generated Samples')

    plt.show()
