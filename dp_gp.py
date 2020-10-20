from qff.embedding import HermiteEmbedding
import numpy as np
from functools import partial
import random


def normalize(v, p=2):
    ''' project vector on to unit L-p ball. '''
    norm=np.linalg.norm(v, ord=p)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def sqexp_p_vectors(x, y, p=2, scale=1.0):
    ''' compute rbf kernel between two vectors x and y '''
    
    d = np.sum(np.power(np.abs(x - y), p), axis=1)
    d = np.expand_dims(np.exp(-d), axis=1)

    return d


INFINITE_KERNELS = ['rbf']


class BO:
    ''' Wrapper over various kernel bayesian optimization matrices.'''
    
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
            self.K_t = None
            self.X_t = None
        
        else:
            self.m = m
            self.feat_func = lambda x: x
            self.inf_kernel = False
            self.rho = 1.0

            self.S_t = self.lam * np.eye(m)
            self.u_t = np.zeros((1, m))
    
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

        if self.X_t is None:
            self.X_t = phi_t
            self.y_t = np.expand_dims(y_t, 0)
        else:
            self.X_t = np.concatenate((self.X_t, x_t), axis=1)
            self.y_t = np.concatenate((self.y_t, y_t), axis=1)
        
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
            mu, S_inv = self._params_fin()

        for x in D_t:
            if self.inf_kernel:
                mu, sigma = self._params_inf(x)
            else:
                phi = self.feat_func(x)
                sigma = np.matmul(phi.T, np.matmul(S_inv, phi))
            
            mus.append(mu)
            sigmas.append(sigma)

        return mus, sigmas


class Env:

    def __init__(self, n=5, d=5, B=1.0, kernel='rbf', noise='normal'):
        
        self.n = n
        self.d = d
        self.B = B

        # initialize kernel parameters
        self.kernel = kernel
        self.init_kernel()

        # initialize noise parameters
        self.noise = noise
        self.init_noise()

        self.actions = []

        # initialize monitor
        self.regrets = []

    
    def init_kernel(self):
        ''' initialize the index set for kernel'''

        self.n_index_set = 4
        if self.kernel == 'linear':
            self.n_index_set = 1

        _alpha = np.random.random_sample((self.n_index_set, ))
        self.alpha = normalize(_alpha, p=1)

        _x = np.random.random_sample((self.n_index_set, self.d))
        _norms = np.linalg.norm(_x, ord=2, axis = 1, keepdims = True)
        self.index_set = _x * self.B / _norms

        if self.kernel == 'rbf':
            self.rbf_scale = 1.0

    def init_noise(self):

        if self.noise == 'normal':
            self.rho = 1.0
    
    def f(self, x):
        ''' get dot product with f'''
        _x_mat = np.tile(x, [self.n_index_set, 1])

        if self.kernel == 'rbf':
            # squared-exponential kernel
            return np.dot(self.alpha, sqexp_p_vectors(self.index_set, _x_mat, p=2, scale=self.rbf_scale))
        
        if self.kernel == 'exp_dist':
            # exponential distance kernel
            _d = np.sqrt(2 - np.einsum('ij,ij->i', self.index_set, _x_mat)) / self.rbf_scale
            _k = np.exp(_d)
            
            return np.dot(self.alpha, _k)
        
        if self.kernel == 'linear':
            # linear kernel
            return np.dot(self.alpha, np.einsum('ij,ij->i', self.index_set, _x_mat))
        
        return 0

    def get_action_set(self):

        actions = []

        for _ in range(self.n):
            # randomly sample d-dimensional vector
            x_i = np.random.random_sample((1, self.d))
            # normalize to l2 ball
            x_i = normalize(x_i, p=2)
            actions.append(x_i)
        

        # calculate best action and store latest r*
        round_rewards = [self.f(x) for x in actions]
        self.opt_x = np.argmax(round_rewards)
        self.opt_r = round_rewards[self.opt_x]
        self.actions = actions

        return actions
    
    def sample_noise(self):

        if self.noise == 'normal':
            return np.random.normal(scale=self.rho)
    
    def play(self, x_t):
        f_x = self.f(x_t)

        y_t = f_x + self.sample_noise()
        r_t = self.opt_r - f_x

        return y_t, r_t


class Agent:

    def __init__(self):
        
        self.t = 0
        pass

    def select_action(self, *args):
        self.t += 1
    

class Random(Agent):

    def __init__(self):
        super(Random, self).__init__()
    
    def select_action(self, D_t):
        # randomly select action
        super(Random, self).select_action()
        return random.choice(D_t)


class GP_UCB(Agent):
    ''' Implements regular GP-UCB with the original kernel. '''

    def __init__(self, lam=1.0, m=0, kernel_type='rbf'):
        
        super(GP_UCB, self).__init__()
        self.bo = BO(kernel_type, m, lam)
        self.delta = 0.01
    
    def update(self, x_t, y_t):
        self.bo.update(x_t, y_t)
    
    def get_beta(self, D_t):
        return np.sqrt(2 * np.log(len(D_t) * (self.bo.t ** 2) * (np.pi ** 2) / (6 * self.delta)))
    
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