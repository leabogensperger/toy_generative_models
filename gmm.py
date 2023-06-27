import numpy as np
import matplotlib.pyplot as plt
import torch

import base_model

class GMM(base_model.BaseModel):
    def __init__(self, n_comp, device, min_covar=1e-6, params=None):
        self.device = device
        self.n_comp = n_comp
        self.sig_stability = min_covar*torch.eye(2).to(device)

        # initialize GMM parameters
        if params is not None:
            self.pi = params[0]
            self.mu = params[1]
            self.sig = params[2]
        else:
            self.pi = torch.ones((n_comp))/n_comp
            self.mu = torch.randn((n_comp, 2))
            self.sig = (torch.eye(2)[None,...]).repeat(n_comp,1,1)

        self.pi, self.mu, self.sig = self.pi.to(device), self.mu.to(device), self.sig.to(device)

    def sample(self, n_samples):
        sel_comp = np.random.choice(a=self.n_comp, size=n_samples, p=self.pi.cpu().numpy())
        L = torch.linalg.cholesky(self.sig + self.sig_stability[None,...])
        return self.mu[sel_comp] + torch.einsum('bij,bj->bi',L[sel_comp],torch.randn(n_samples,2,device=self.device))

    def pdf(self, x, mu, sig):
        # evaluate multivariate probability density function for a Gaussian 
        fact = 1/(torch.sqrt(torch.linalg.det(sig + self.sig_stability))*2*np.pi)
        return fact*torch.exp(-0.5*torch.einsum('bj,bj->b', (x - mu), torch.einsum('ij,bi->bj',torch.linalg.inv(sig+self.sig_stability),(x - mu))))

    def fit(self, x, max_iter, eps=1e-6):
        S = x.shape[0]
        neglogllh = [] # track negative log-likelihood

        for iter in range(max_iter):
            # compute likelihood
            likelihood = torch.zeros((S, self.n_comp)).to(x.device)         
            for comp in range(self.n_comp):
                likelihood[:,comp] = self.pdf(x, self.mu[comp],self.sig[comp])
            
            # Expectation: re-compute responsibilites
            numerator = likelihood * self.pi
            denominator = numerator.sum(axis=1)[:, np.newaxis]
            w = numerator / denominator          

            # Maximization: re-compute GMM parameters
            N = w.sum(0)
            self.mu = (w[:,:,None]*x[:,None,:]).sum(0)/N[:,None]

            xh = x[:,None,:] - self.mu[None,...]
            self.sig = torch.einsum('bki,bkj->kij',torch.einsum('bk,bkj->bkj',w,xh),xh)/N[:,None,None]
            self.pi = N/(x.shape[0])

            # logging negative log-likelihood
            neglogllh.append(-torch.log(numerator.sum(-1)).sum().item())

            # check stopping criterion
            if iter >= 2 and (np.abs(neglogllh[-1]-neglogllh[-2]) <= eps):
                print('\nGMM: stopping criterion evoked in iteration %i' %iter)
                break

        return neglogllh

def estimate_density(x, cfg):
    gmm = GMM(n_comp=cfg.n_comp, device=x.device)
    neglogllh = gmm.fit(x=x, max_iter = cfg.max_iter, eps=cfg.eps)

    # plot fitted GMM
    min_val, max_val = x.min(), x.max()
    yy, xx = torch.meshgrid(torch.linspace(min_val,max_val,100),torch.linspace(min_val,max_val,100)) 
    x_test = torch.stack((xx.ravel(),yy.ravel())).T.to(x.device)

    # evaluate gmm on test data
    likelihood = torch.zeros((x_test.shape[0], gmm.n_comp)).to(x.device)
    for comp in range(gmm.n_comp):
        likelihood[:,comp] = gmm.pdf(x_test.to(x.device),gmm.mu[comp],gmm.sig[comp])
    numerator = likelihood * gmm.pi
    f = numerator.sum(-1)    

    # sample from gmm
    samples = gmm.sample(cfg.n_samples)


    # plotting
    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    gs = fig.add_gridspec(2,2)

    # neg log likelihood plot
    f_ax11 = fig.add_subplot(gs[0, 0])
    f_ax11.set_title('Negative Log-Likelihood'), f_ax11.set_xlabel('EM iterations')
    f_ax11.plot(neglogllh)

    # plot estimated density
    f_ax12 = fig.add_subplot(gs[0, 1])
    f_ax12.contourf(x_test[:,0].cpu().reshape(xx.shape[0],xx.shape[1]),x_test[:,1].cpu().reshape(xx.shape[0],xx.shape[1]),\
        f.cpu().reshape(xx.shape[0],xx.shape[1]),cmap='viridis',levels=20)
    f_ax12.plot(x[:,0].cpu(),x[:,1].cpu(),'.',alpha=0.2)
    f_ax12.set_title('Estimated Density')

    # plot generated samples
    f_ax21 = fig.add_subplot(gs[1, 0], sharex=f_ax12, sharey=f_ax12)
    f_ax21.scatter(x[:,0].cpu(),x[:,1].cpu(),marker='.'), f_ax21.set_title('Original Data Samples')
    f_ax22 = fig.add_subplot(gs[1, 1], sharex=f_ax12, sharey=f_ax12)
    f_ax22.scatter(samples[:,0].cpu(),samples[:,1].cpu(),marker='.'), f_ax22.set_title('Generated Samples')
    plt.show()