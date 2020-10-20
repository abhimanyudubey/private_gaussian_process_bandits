import numpy as np

from .helpers import sqexp_p_vectors, normalize

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