import numpy as np
from functools import partial
import random

from .helpers import sqexp_p_vectors
from .qff.embedding import HermiteEmbedding



INFINITE_KERNELS = ['rbf']


class BO:
    ''' Wrapper over various kernel bayesian optimization methods.'''
    
    def __init__(self, kernel='rbf', m=0, lam=1.0):
        
        self.kernel = kernel
        self.inf_kernel = False
        self.lam = lam
        self.t = 0

        if self.kernel not in INFINITE_KERNELS and m<1:
            raise ValueError('Incorrect m provided')
            
        if m == 0:
            self.k_func = partial(sqexp_p_vectors, p=2, scale=1.0)
            self.inf_kernel = True
        
        else:
            self.m = m
            self.feat_func = lambda x: x
            self.inf_kernel = False
            self.rho = 1.0
        
        self.reset()

    
    def reset(self):
        ''' Reset the BO parameters '''
        self.t = 0
        if self.inf_kernel:
            self.K_t = None
            self.X_t = None
        
        else:
            self.S_t = self.lam * np.eye(self.m)
            self.u_t = np.zeros((1, self.m))
            self.X_t = None
    
    
    def _update_inf_dim(self, x_t, y_t):
        # rank 1 update of kernel matrices

        if self.K_t is None:
            self.K_t = 1.0/(self.k_func(x_t, x_t) + self.lam) * np.ones((1, 1))
        else:
            inp_X = np.tile(x_t, [self.t, 1])
            b = self.k_func(inp_X, self.X_t)
            
            K_22 = np.power(self.k_func(x_t,x_t) + self.lam - np.matmul(b.T, np.matmul(self.K_t, b)), -1)
            K_11 = self.K_t + K_22 * np.matmul(self.K_t, np.matmul(b, np.matmul(b.T, self.K_t.T)))
            K_12 = -K_22 * np.matmul(self.K_t, b)
            K_21 = -K_22 * np.matmul(b.T, self.K_t.T)
            
            K_t_up = np.concatenate((K_11, K_12), axis=1)
            K_t_down = np.concatenate((K_21, K_22), axis=1)

            self.K_t = np.concatenate((K_t_up, K_t_down), axis=0)
        
        self.t += 1
        # padding y appropriately
        y_t = y_t * np.ones((1, 1))
        
        if self.X_t is None:
            self.X_t = x_t
            self.y_t = y_t
        else:
            self.X_t = np.concatenate((self.X_t, x_t), axis=0)
            self.y_t = np.concatenate((self.y_t, y_t), axis=0)
    
    def _update_fin_dim(self, x_t, y_t):
        # regular finite-dimensional update
        self.t += 1

        phi_t = self.feat_func(x_t)
        y_t = np.expand_dims(y_t, 0)

        if self.X_t is None:
            self.X_t = phi_t
            self.y_t = y_t
        else:
            self.X_t = np.concatenate((self.X_t, x_t), axis=0)
            self.y_t = np.concatenate((self.y_t, y_t), axis=0)
        
        self.S_t += np.matmul(phi_t, phi_t.T)
        self.u_t += y_t * phi_t
    
    def update(self, x_t, y_t):
        ''' Update internal parameters with new observations.'''
        if self.inf_kernel:
            self._update_inf_dim(x_t, y_t)
        else:
            self._update_fin_dim(x_t, y_t)
    
    def _params_fin(self):
        # return mu, sigma for finite k
        S_inv = np.linalg.inv(self.S_t)
        mu = np.matmul(S_inv, np.matmul(self.X_t.T, self.y_t))

        return mu, S_inv
        
    def _params_inf(self, x):
        # return mu, sigma for infinite dimensional k
        x_rep = np.tile(x, [self.t, 1])
        k_t = self.k_func(self.X_t, x_rep)
        
        mu = np.matmul(k_t.T, np.matmul(self.K_t, self.y_t))
        sigma = np.sqrt(self.k_func(x, x) - np.matmul(k_t.T, np.matmul(self.K_t, k_t)))

        return mu, sigma
    
    def get_posterior(self, D_t):
        ''' Compute the posterior mean and variance for each arm in D_t '''
        
        mus, sigmas = [], []
        if not self.inf_kernel:
            u, S_inv = self._params_fin()

        for x in D_t:
            if self.inf_kernel:
                mu, sigma = self._params_inf(x)
            else:
                phi = self.feat_func(x)
                mu = np.dot(phi, u)
                sigma = np.matmul(phi, np.matmul(S_inv, phi.T))
            
            mus.append(mu)
            sigmas.append(sigma)

        return mus, sigmas


class Agent:

    def __init__(self):
        
        self.t = 0
        pass

    def select_action(self, *args, **kwargs):
        self.t += 1
    
    def update(self, *args, **kwargs):
        pass

    def reset(self):
        pass

class Random(Agent):

    def __init__(self):
        super(Random, self).__init__()
    
    def select_action(self, D_t, *args, **kwargs):
        # randomly select action
        super(Random, self).select_action()
        return random.choice(D_t)


class GP_UCB(Agent):
    ''' Implements regular GP-UCB with the original kernel. '''

    def __init__(self, lam=1.0, m=0, B=1.0, kernel='rbf'):
        
        super(GP_UCB, self).__init__()
        self.kernel = kernel
        self.bo = BO(kernel, m, lam)
        self.delta = 0.01
        self.B = B
    
    def update(self, x_t, y_t):
        self.bo.update(x_t, y_t)
    
    def reset(self):
        self.bo.reset()
    
    def get_beta(self, D_t):

        if self.kernel in INFINITE_KERNELS:
            return np.sqrt(2 * np.log(len(D_t) 
                * (self.bo.t ** 2) * (np.pi ** 2) / (6 * self.delta)))
        else:
            # finite dimensional kernel, use Abbasi-Yadkori
            return self.bo.rho * \
                np.sqrt(self.bo.m * np.log(1 + self.bo.t * (self.B**2) / self.bo.lam) 
                - np.log(self.delta)) + self.bo.lam ** (0.5) * self.B
    
    def select_action(self, D_t, beta_mult):
        
        if self.bo.t == 0:
            return random.choice(D_t)
        
        beta_t = self.get_beta(D_t) * beta_mult
        best_x, max_ucb = None, -np.inf
        mus, sigmas = self.bo.get_posterior(D_t)
        for x, mu, sigma in zip(D_t, mus, sigmas):
            ucb_x  = mu + beta_t * sigma
            if max_ucb < ucb_x:
                best_x, max_ucb = x, ucb_x

        return best_x