import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.optim import Adam

import base_model

class DSM(base_model.BaseModel):
    def __init__(self, sigma_min, sigma_max, L, size_hidden, device):
        super().__init__()
        self.device = device
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.L = L # number of noise levels
        self.sigmas_all = self.gen_sigmas().to(self.device)

        self.W1 = nn.Linear(3, size_hidden, bias=True) # additionally condition on noise level
        self.W2 = nn.Linear(size_hidden, size_hidden, bias=True)
        self.W3 = nn.Linear(size_hidden, 2, bias=True)

    def gen_sigmas(self):
        return torch.tensor(np.exp(np.linspace(np.log(self.sigma_min),np.log(self.sigma_max), self.L))).type(torch.float32)

    def forward(self, ipt, sigma_):
        ipt_stacked = torch.hstack((ipt, sigma_))
        tmp = torch.relu(self.W1(ipt_stacked))
        tmp = torch.relu(self.W2(tmp))
        return self.W3(tmp)

    def fit(self, x, max_iter, lr):
        optim = Adam(params=self.parameters(), lr=lr)
        self.train()
        
        losses = []
        for iter_ in range(max_iter):
            # sample noise levels (disc. way here, must also work continuously)
            idx = torch.randint(0, self.L, (x.shape[0],))
            used_sigmas = (self.sigmas_all[idx][:,None])

            # add noise
            z = x + used_sigmas*torch.randn_like(x)
            
            # use Tweedie's formula for single step denoising loss
            target = 1/(used_sigmas**2)*(x-z)
            scores = self.forward(z,used_sigmas)
            loss = (used_sigmas**2*((target-scores)**2)).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            x = shuffle(x)
        
        return losses
    
    def sample(self, n_samples):
        # sampling - PC scheme
        samples = self.sigma_max*torch.randn(n_samples,2,device=self.device)
        sigma_t_prev = self.sigmas_all[-1]
        samples_all = []
        samples_all.append(samples.detach().clone())

        self.eval()
        with torch.no_grad():
            for i, sigma_t in enumerate(torch.flip(self.sigmas_all,dims=(0,))[1::]):
                scores = self(samples,sigma_t_prev*torch.ones((samples.shape[0],1),device=self.device))
                z = torch.randn(n_samples,2,device=self.device)
                tau = np.abs(sigma_t_prev.item()**2 - sigma_t.item()**2)

                # predictor step
                samples += tau*scores + np.sqrt(tau)*z
                sigma_t_prev = sigma_t.clone()

                samples_all.append(samples.detach().clone())
                
        return samples_all

def estimate_density(x, cfg):
    dsm = DSM(sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max, L=cfg.L, size_hidden=cfg.size_hidden, device=x.device)
    dsm = dsm.to(x.device)
    print('DSM: #learnable params=%d' %(dsm.count_parameters()))

    # test min/max noise levels
    x_noise_min = x + cfg.sigma_min*torch.randn_like(x)
    x_noise_max = x + cfg.sigma_max*torch.randn_like(x)

    loss = dsm.fit(x,max_iter=cfg.max_iter,lr=cfg.lr)

    # sampling
    samples = dsm.sample(n_samples=cfg.n_samples)

    # plotting
    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    gs = fig.add_gridspec(3,3)

    # plot sigma_min/sigma_max
    f_ax11 = fig.add_subplot(gs[0, 0])
    f_ax11.scatter(x_noise_min[:,0].cpu(),x_noise_min[:,1].cpu(),marker='.'), f_ax11.set_title(r'Data $x$ with $\sigma_{\min}=%.4f$' %cfg.sigma_min)
    f_ax12 = fig.add_subplot(gs[0, 1],sharex=f_ax11, sharey=f_ax11)
    f_ax12.scatter(x_noise_max[:,0].cpu(),x_noise_max[:,1].cpu(),marker='.'), f_ax12.set_title(r'Data $x$ with $\sigma_{\max}=%.4f$' %cfg.sigma_max)

    # plot loss
    f_ax13 = fig.add_subplot(gs[0, 2])
    f_ax13.set_title('Training loss'), f_ax13.set_xlabel('iterations')
    f_ax13.plot(np.asarray(loss))

    # plot learned scores
    res = 100
    xx,yy = np.meshgrid(np.linspace(x.min().item(),x.max().item(),res),np.linspace(x.min().item(), x.max().item(),res))
    positions = torch.tensor(np.vstack([xx.ravel(), yy.ravel()])).type(torch.float32).to(dsm.device).T
    noiselevels = [cfg.sigma_min, cfg.sigma_max]

    step = 2 # step for plotting the scores
    for idx, noiselevel in enumerate(noiselevels):
        scores = dsm(positions, noiselevel*torch.ones((positions.shape[0],1)).to(dsm.device))

        U = scores[:,0].reshape(res,res).detach().cpu().numpy()
        V = scores[:,1].reshape(res,res).detach().cpu().numpy()
        M = np.hypot(U, V)

        f_ax2x = fig.add_subplot(gs[1,idx])
        f_ax2x.quiver(xx[::step,::step],yy[::step,::step],\
            U[::step,::step],V[::step,::step],M[::step,::step])
        f_ax2x.set_title(r'Scores for $\sigma=%.4f$' %noiselevel)

    # plot samples
    f_ax31 = fig.add_subplot(gs[2, 0])
    f_ax31.scatter(x[:,0].cpu(), x[:,1].cpu(), marker='.'), f_ax31.set_title('Original Data Samples')

    f_ax32 = fig.add_subplot(gs[2, 1], sharex=f_ax31, sharey=f_ax31)
    f_ax32.scatter(samples[-1][:,0].cpu(), samples[-1][:,1].cpu(), marker='.'), f_ax32.set_title('Generated Samples')

    plt.show()
