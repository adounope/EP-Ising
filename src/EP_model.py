import numpy as np
import openjij.cxxjij as oj
import src.utils as utils


class EP_model:
    def __init__(self, input_size=0, hidden_size=0, output_size=0, N_spins=0, J_connected=0,\
                 N_neal_temps=(2**6, 2**6), N_neal_steps=(2**4, 2**4), neal_start_temp=(2**6, 2**3),\
                 output_duplicate=4, lr = 1/4, beta=8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
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
        self.W = np.random.randn(self.input_size, self.hidden_size)
        #bias not needed since it already exists as ext field of input spins

    def create_schedule(self):
        self.neal_schedule_list = oj.utility.make_classical_schedule_list(1/self.neal_start_temp, 2**10, self.N_neal_steps, self.N_neal_temps)
        self.rev_neal_schedule_list = oj.utility.make_classical_schedule_list(1/self.rev_neal_start_temp, 2**10, self.N_rev_neal_steps, self.N_rev_neal_temps)

    def select_IO(self):
        # choose lattice site to use as input and output
        self.ising_hidden_idx = np.arange(self.hidden_size)
        self.ising_out_idx = np.arange(self.N_spins - self.output_size, self.N_spins)
    
    def forward(self, X: np.ndarray):
        '''
        X: <np.array> batch, 196
        max batch size = self.max_process
        '''
        if len(X) > self.max_process:
            print('batch size too large')
            exit()
        cfg = self._forward_single(X[0])

        return self.label_to_num(y_raw=cfg[self.ising_out_idx][None, :]), cfg[None, :]
    


    def _forward_single(self, X: np.ndarray, beta=0, Y=0, lattice=0):
        '''
        X: <np.array> 196
        Y: <np.array> output_size (only when nudge neal)
        beta: output clamping strength
        lattice: <np.array> free phase of lattice (only when nudge neal)
        '''
        nudge = (beta!=0)
        H = self.H.copy()
        H[self.ising_out_idx] += -beta * Y # minus sign, since energy is inverted
        H[self.ising_hidden_idx] += X@self.W
        J = utils.convert_to_jijDense(self.J, H)
        if nudge:
            schedule_list = self.rev_neal_schedule_list
        else:
            lattice = J.gen_spin()
            schedule_list = self.neal_schedule_list
        ising_model = oj.system.make_classical_ising(lattice, J)
        oj.algorithm.Algorithm_SingleSpinFlip_run(ising_model, schedule_list)
        config = np.array(oj.result.get_solution(ising_model))
        #y = config[self.ising_out_idx]
        return config
    
    def gradient(self, X, lattice_free, lattice_nudge): # batch_size, N_spin
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
        dW = X.T @ (lattice_free[:, self.ising_hidden_idx] - lattice_nudge[:, self.ising_hidden_idx])

        return {'dJ':dJ, 'dH': dH, 'dW':dW}
    
    # k N + R // N = k, max: i = k-1, N * (k-1) : N * (k)
    def train(self, X: np.ndarray, Y: np.ndarray):
        '''
        batch, 196
        max batch size = self.max_process
        '''
        batch_size = len(X)
        if batch_size > self.max_process:
            print('batch size too large')
            exit()
        pred = np.empty((batch_size), int)

        def for_back(X, beta, Y):
            lattice_free = self._forward_single(X)
            if (lattice_free[self.ising_out_idx]==Y).all(): #skip if free phase is optimal
                return (lattice_free, lattice_free) # this way have no net gradient produced
                
            lattice_nudge = self._forward_single(X, beta, Y, lattice_free)
            return (lattice_free, lattice_nudge)

        lattices_free = np.empty((batch_size, self.N_spins), dtype=np.int8)
        lattices_nudge = np.empty((batch_size, self.N_spins), dtype=np.int8)


        lattices_free[0], lattices_nudge[0] = for_back(X[0], self.beta, Y[0])

        pred = self.label_to_num(y_raw=lattices_free[:, self.ising_out_idx])
        #nudge_pred = self.convert_output_to_idx(y_raw=ys_n)
        # F = E + βC  (nudged version of E)
        grad = self.gradient(X, lattices_free, lattices_nudge)
        scale = self.lr / self.beta
        self.J += scale * grad['dJ']
        self.H += scale * grad['dH']
        self.W += scale * grad['dW']
        # destabalize E (free phase) and stabalize F (nudge phase)
        return pred#, lattice_change, y_change, nudge_pred 
    
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
        self.W = np.load(f'{foldername}/weights/{best_epoch}_W.npy')


