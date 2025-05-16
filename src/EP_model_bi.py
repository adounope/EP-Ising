import numpy as np
import openjij.cxxjij as oj
import src.utils as utils


class EP_model:
    def __init__(self, input_size=0, hidden_size=0, output_size=0, N_spins=0, J_connected=0,\
                 N_neal_temps=(2**6, 2**6), N_neal_steps=(2**4, 2**4), neal_start_temp=(2**6, 2**3),\
                 output_duplicate=4, lr = 1/4, beta=8):
        self.input_size = self.i_s = input_size
        self.hidden_size = self.h_s = hidden_size
        self.output_size = self.o_s = output_size
        self.N_spins = N_spins
        self.J_connected = J_connected

        self.init_params()
        

        self.N_neal_temps, self.N_rev_neal_temps = N_neal_temps
        self.N_neal_steps, self.N_rev_neal_steps = N_neal_steps
        self.neal_start_temp, self.rev_neal_start_temp = neal_start_temp

        self.max_process = 1
        self.output_duplicate = output_duplicate
        self.lr = lr
        self.beta = beta # nudge strength of EP

        self.select_IO()

    def init_params(self):
        np.random.seed(80805)
        tmp = np.triu(np.random.randn(self.N_spins, self.N_spins) * self.J_connected)
        self.J = tmp + tmp.T #masked random symmetric
        self.H = np.zeros(self.N_spins) # works as bias
        #self.W = np.random.randn(self.input_size, self.hidden_size)
        #bias not needed since it already exists as ext field of input spins

    def create_schedule(self):
        self.neal_schedule_list = oj.utility.make_classical_schedule_list(1/self.neal_start_temp, 2**10, self.N_neal_steps, self.N_neal_temps)
        self.rev_neal_schedule_list = oj.utility.make_classical_schedule_list(1/self.rev_neal_start_temp, 2**10, self.N_rev_neal_steps, self.N_rev_neal_temps)

    def select_IO(self):
        # choose lattice site to use as input and output
        self.ising_in_idx = np.arange(self.input_size)
        self.ising_hidden_idx = np.arange(self.input_size, self.hidden_size)
        self.ising_out_idx = np.arange(self.N_spins - self.output_size, self.N_spins)
    
    def forward(self, X: np.ndarray):
        '''
        X: <np.array> batch, 196
        max batch size = self.max_process
        '''
        if len(X) > self.max_process:
            print('batch size too large')
            exit()
        cfg, _ = self._forward_single(X[0])

        return self.label_to_num(y_raw=cfg[self.ising_out_idx][None, :]), cfg[None, :]
    


    def _forward_single(self, X=0, beta=0, Y=0):
        '''
        X: <np.array> 196
        Y: <np.array> output_size (only when nudge neal)
        beta: output clamping strength
        lattice: <np.array> free phase of lattice (only when nudge neal)
        '''
        config = np.zeros_like(self.H, dtype=np.int8)
        np_H = self.H.copy()
        # clamp input
        np_H[ self.i_s : ] += X @ self.J[ 0 : self.i_s, self.i_s : ]
        J = utils.convert_to_jijDense(self.J[ self.i_s : , self.i_s : ], np_H[ self.i_s : ]) # select only needed spins
        lattice = J.gen_spin()
        
        ising_model = oj.system.make_classical_ising(lattice, J)
        oj.algorithm.Algorithm_SingleSpinFlip_run(ising_model, self.neal_schedule_list)

        lattice = oj.result.get_solution(ising_model)
        config[ self.i_s : ] = np.array(lattice)
        if beta == 0 or (config[self.ising_out_idx]==Y).all():
            return config, config
        # nudge output
        # minus sign, since energy is inverted
        np_H[self.ising_out_idx] -= beta * Y 
        J = utils.convert_to_jijDense(self.J[ self.i_s : , self.i_s : ], np_H[ self.i_s : ]) # select only needed spins
        config_nudge = np.zeros_like(self.H, dtype=np.int8)

        ising_model = oj.system.make_classical_ising(lattice, J)
        oj.algorithm.Algorithm_SingleSpinFlip_run(ising_model, self.rev_neal_schedule_list)

        config_nudge[ self.i_s : ] = np.array(oj.result.get_solution(ising_model))
        return config, config_nudge
    
    def _forward_single_reverse(self, X=0, beta=0, Y=0):
        '''
        X: <np.array> 196 (only when nudge neal)
        Y: <np.array> output_size
        beta: output clamping strength
        lattice: <np.array> free phase of lattice (only when nudge neal)
        '''
        config = np.zeros_like(self.H, dtype=np.int8)
        np_H = self.H.copy()
        # clamp input
        np_H[ : -self.o_s ] += Y @ self.J[ -self.o_s :, : -self.o_s ]
        J = utils.convert_to_jijDense(self.J[ : -self.o_s, : -self.o_s ], np_H[ : -self.o_s ]) # select only needed spins
        lattice = J.gen_spin()

        ising_model = oj.system.make_classical_ising(lattice, J)
        oj.algorithm.Algorithm_SingleSpinFlip_run(ising_model, self.neal_schedule_list)

        lattice = oj.result.get_solution(ising_model)
        config[ : -self.o_s ] = np.array(lattice)
        if beta == 0 or (config[self.ising_in_idx]==X).all():
            return config, config
        # nudge output
        # minus sign, since energy is inverted
        np_H[self.ising_in_idx] -= beta * X
        J = utils.convert_to_jijDense(self.J[ : -self.o_s, : -self.o_s ], np_H[ : -self.o_s ]) # select only needed spins
        config_nudge = np.zeros_like(self.H, dtype=np.int8)

        ising_model = oj.system.make_classical_ising(lattice, J)
        oj.algorithm.Algorithm_SingleSpinFlip_run(ising_model, self.rev_neal_schedule_list)

        config_nudge[ : -self.o_s ] = np.array(oj.result.get_solution(ising_model))
        return config, config_nudge
    def gradient(self, lattice_free, lattice_nudge): # batch_size, N_spin
        # dEdJ = (lattice_free.T @ lattice_free) * self.J_connected #N_spins, N_spins
        # dEdH = lattice_free # batch_size, N_spins
        # dEdW = X.T @ dEdH[:, self.ising_hidden_idx]
        # dEdH = dEdH.sum(axis=0)

        # dFdJ = (lattice_nudge.T @ lattice_nudge) * self.J_connected #N_spins, N_spins
        # dFdH = lattice_nudge # batch_size, N_spins
        # dFdW = X.T @ dFdH[:, self.ising_hidden_idx]
        # dFdH = dFdH.sum(axis=0)

        dJ = ((lattice_free.T @ lattice_free) - (lattice_nudge.T @ lattice_nudge)) * self.J_connected
        dH = (lattice_free - lattice_nudge).sum(axis=0)
        # dW = X.T @ (lattice_free[:, self.ising_hidden_idx] - lattice_nudge[:, self.ising_hidden_idx])

        return {'dJ':dJ, 'dH': dH}#, 'dW':dW}
    
    # k N + R // N = k, max: i = k-1, N * (k-1) : N * (k)
    def train(self, X: np.ndarray, Y: np.ndarray, mode='bi'):
        '''
        1, 196
        max batch size = self.max_process
        mode: control whether training is forward, reverse, bidirectional
        '''

        #X = np.tile(X, (1, self.input_duplicate)) # 1, 196 * 4
        #Y = np.tile(Y, (1, self.output_duplicate))
        batch_size = len(X)
        if batch_size > self.max_process:
            print('batch size too large')
            exit()
        #pred = np.empty((batch_size), int)

        tmp = 1
        if mode == 'bi': # in case of bidirectional training
            tmp = 2
        lattices_free = np.zeros((tmp, self.N_spins), dtype=np.int8)
        lattices_nudge = np.zeros((tmp, self.N_spins), dtype=np.int8)


        if mode == 'f' or mode == 'bi':
            lattices_free[0], lattices_nudge[0] = self._forward_single(X[0], self.beta, Y[0])
        if mode == 'r' or mode == 'bi':
            lattices_free[-1], lattices_nudge[-1] = self._forward_single_reverse(X[0], self.beta, Y[0])


        pred = self.label_to_num(y_raw=lattices_free[0, self.ising_out_idx][None, :])
        reconstruction_loss = (X[0] != lattices_free[-1, self.ising_in_idx]).mean()
        # F = E + βC  (nudged version of E)
        grad = self.gradient(lattices_free, lattices_nudge)
        scale = self.lr / self.beta
        self.J += scale * grad['dJ']
        self.H += scale * grad['dH']
        # self.W += scale * grad['dW']
        # destabalize E (free phase) and stabalize F (nudge phase)
        return pred, reconstruction_loss
    
    def label_to_num(self, y_raw):
        tmp = y_raw.reshape(len(y_raw), self.output_size//self.output_duplicate, -1).mean(axis=-1) #batch, 10
        tmp = (tmp == np.max(tmp, axis=1)[:, None]) #batch, 10 (bool)
        invalid = (tmp.sum(axis=1) != 1) # batch (detect case of multiple maximum)
        y = np.argmax(tmp, axis=1)
        y[invalid] = -1
        return y

    def load_trained_param(self, foldername, best_epoch):
        self.J = np.load(f'{foldername}/weights/{best_epoch}_J.npy')
        self.H = np.load(f'{foldername}/weights/{best_epoch}_H.npy')
        #self.W = np.load(f'{foldername}/weights/{best_epoch}_W.npy')


