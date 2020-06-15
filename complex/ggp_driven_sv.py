import sys
sys.path.append('../')
import argparse
import os
import pickle
import numpy as np
import numpy.random as npr
import numba as nb
import particles.distributions as dists
import matplotlib.pyplot as plt

from utils import logit, sigmoid, gammaln
from gbfry import GGPsumrnd
from levy_driven_sv import LevyDrivenSV

# x: 2 dim vector, x[0] = intergrated volatility bar V, x[1] = volatility V
@nb.njit(nb.f8[:,:](nb.f8, nb.f8, nb.f8, nb.f8, nb.f8[:]), fastmath=True)
def volatility_transition(eta, sigma, c, lam, vp):
    K_hat = 30
    N = len(vp)
    x = np.zeros((N, 2))
    for i in range(N):
        x[i,1] = np.exp(-lam)*vp[i]
        dz = 0
        # GGP part, truncated upto K_hat terms
        xi = 0
        for j in range(K_hat):
            xi += npr.exponential(1)
            log_w = min(
                    -(np.log(xi) + sigma*np.log(c) + gammaln(1-sigma) - np.log(eta*lam))/sigma,
                    np.log(npr.exponential(1./c)) + np.log(npr.rand())/sigma)
            w = np.exp(log_w)
            x[i,1] += w*np.exp(-lam*npr.rand())
            dz += w

        # finite-activity part
        K = npr.poisson(eta*lam)
        for j in range(K):
            w = npr.gamma(1-sigma, 1/c)
            x[i,1] += w*np.exp(-lam*npr.rand())
            dz += w
        x[i,0] = (vp[i] - x[i,1] + dz)/lam
    return x

class PX0(dists.ProbDist):
    def __init__(self, eta, sigma, c, lam):
        self.eta = eta
        self.sigma = sigma
        self.c = c
        self.lam = lam

    def rvs(self, size=1, n_warmup=1):
        v0 = GGPsumrnd(self.eta/self.c**self.sigma, self.sigma, self.c, size)
        for _ in range(n_warmup-1):
            v0 = GGPsumrnd(self.eta/self.c**self.sigma, self.sigma, self.c, size)
            v0 = v0[...,1]
        return volatility_transition(self.eta, self.sigma, self.c, self.lam, v0)

class PX(dists.ProbDist):
    def __init__(self, eta, sigma, c, lam, xp):
        self.eta = eta
        self.sigma = sigma
        self.c = c
        self.lam = lam
        self.vp = xp[...,1]

    def rvs(self, size=1):
        return volatility_transition(self.eta, self.sigma, self.c, self.lam, self.vp)

class GGPDrivenSV(LevyDrivenSV):

    params_name = {
            'log_eta':'eta',
            'logit_sigma':'sigma',
            'log_c':'c',
            'log_lam':'lambda'}

    params_latex = {
            'log_eta':'\eta',
            'logit_sigma':'\sigma',
            'log_c':'c',
            'log_lam':'\lambda'}

    params_transform = {
            'log_eta':np.exp,
            'logit_sigma':sigmoid,
            'log_c':np.exp,
            'log_lam':np.exp}

    def __init__(self, mu=0.0, beta=0.0,
            log_eta=np.log(4.0), logit_sigma=logit(0.3),
            log_c=np.log(1.0), log_lam=np.log(0.01)):
        super(GGPDrivenSV, self).__init__(
                mu=mu,
                beta=beta,
                log_lam=log_lam)
        self.log_eta = log_eta
        self.logit_sigma = logit_sigma
        self.log_c = log_c

    def PX0(self):
        eta = np.exp(self.log_eta)
        sigma = sigmoid(self.logit_sigma)
        c = np.exp(self.log_c)
        lam = np.exp(self.log_lam)
        return PX0(eta, sigma, c, lam)

    def PX(self, t, xp):
        eta = np.exp(self.log_eta)
        sigma = sigmoid(self.logit_sigma)
        c = np.exp(self.log_c)
        lam = np.exp(self.log_lam)
        return PX(eta, sigma, c, lam, xp)

    @staticmethod
    def get_prior():
        prior_dict = {
                'log_eta':dists.LogD(dists.Gamma(a=1., b=1.)),
                'logit_sigma':dists.LogitD(dists.Beta(a=1., b=1.)),
                'log_c':dists.LogD(dists.Gamma(a=1., b=1.)),
                'log_lam':dists.LogD(dists.Gamma(a=1., b=1.))}
        return dists.StructDist(prior_dict)

    @staticmethod
    def get_theta0(y):
        prior = GGPDrivenSV.get_prior()
        theta0 = prior.rvs()
        theta0['log_eta'] = np.log(1.0) + 0.1*npr.randn()
        theta0['log_c'] = np.log(1.0) + 0.1*npr.randn()
        theta0['log_lam'] = np.log(0.01) + 0.1*npr.randn()
        return theta0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--eta', type=float, default=4.0)
    parser.add_argument('--sigma', type=float, default=0.2)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()

    data = {}

    true_params = {'mu':0.0,
            'beta':0.0,
            'log_lam':np.log(0.01),
            'log_eta':np.log(args.eta),
            'logit_sigma':logit(args.sigma),
            'log_c':np.log(args.c)}
    model = GGPDrivenSV(**true_params)
    x, y = model.simulate(args.T)
    y = np.array(y)
    x = np.array(x)[:,0,1]

    data['true_params'] = true_params
    data['x'] = x
    data['y'] = y

    if not os.path.isdir('../data'):
        os.makedirs('../data')

    if args.filename is None:
        filename = ('../data/ggp_driven_sv_'
                'T_{}_'
                'eta_{}_'
                'sigma_{}_'
                'c_{}'
                '.pkl'
                .format(args.T, args.eta, args.sigma, args.c))
    else:
        filename = args.filename

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    plt.figure('x')
    plt.plot(x)
    plt.figure('y')
    plt.plot(y)
    plt.show()
