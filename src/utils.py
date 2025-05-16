import numpy as np
import openjij.cxxjij as oj
import pickle
from tqdm import tqdm


def calculus_d1(y, x): #assume x is uniform
    dy = ( np.roll(y, -1) - np.roll(y, 1) )
    dx = x[1] - x[0]
    
    dy[0] = dy[-1] = 0
    return dy/dx

def batch_idx(N, batch_size):
    tmp = np.linspace(0, np.ceil(N/batch_size), int(np.ceil(N/batch_size))+1, endpoint=True, dtype=int) * batch_size
    return [(tmp[i], np.minimum(tmp[i+1], N)) for i in range(len(tmp)-1)]

def convert_to_jijDense(np_J, np_H): # E = Σ_{i<j} J_ij * σ_i * σ_j + Σ_{i} H_i * σ_i
    N = len(np_H)
    tmp = np.zeros((N+1, N+1))
    tmp[-1, -1] = 1
    tmp[:N, :N] = np_J
    tmp[:-1, -1] = np_H
    J = oj.graph.Dense(N)
    J.set_interaction_matrix(tmp)
    return J

def load_data(folder_name):
    x_train = np.load(f'{folder_name}/x_train.npy')
    y_train = np.load(f'{folder_name}/y_train.npy')
    y_train_idx = np.load(f'{folder_name}/y_train_idx.npy')

    x_val = np.load(f'{folder_name}/x_val.npy')
    y_val = np.load(f'{folder_name}/y_val.npy')
    y_val_idx = np.load(f'{folder_name}/y_val_idx.npy')

    return x_train, y_train, y_train_idx, x_val, y_val, y_val_idx

def save_obj(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def load_trained_param(foldername, best_epoch):
    J = np.load(f'{foldername}/weights/{best_epoch}_J.npy')
    H = np.load(f'{foldername}/weights/{best_epoch}_H.npy')
    W = np.load(f'{foldername}/weights/{best_epoch}_W.npy')
    return {'J':J, 'H':H, 'W':W}

def analysis(np_J, np_H, temp: np.ndarray, N_sample: int, burn=0):
    E = np.zeros(len(temp))
    E2 = np.zeros(len(temp))
    M = np.zeros(len(temp))
    M2 = np.zeros(len(temp))
    abs_M = np.zeros(len(temp))

    J = convert_to_jijDense(np_J, np_H)
    ising_model = oj.system.make_classical_ising(J.gen_spin(), J)

    beta_2_idx_table = {1/temp[i]: i for i in range(len(temp))}

    def callback_f(system, beta):
        i = beta_2_idx_table[beta]

        tmp_E = J.calc_energy(system.spin)
        tmp_M = system.spin[:-1].sum()
        E[i] += tmp_E
        E2[i] += tmp_E**2
        M[i] += tmp_M
        abs_M[i] += np.abs(tmp_M)
        M2[i] += (tmp_M)**2

    for t in tqdm(np.flip(temp)):
        #schedule_list = oj.utility.make_classical_schedule_list(2**-5, 1/t, 1, 2**10)
        #oj.algorithm.Algorithm_SingleSpinFlip_run(ising_model, schedule_list)
        oj.algorithm.Algorithm_SingleSpinFlip_run(ising_model, [(1/t, int(burn*N_sample))] ) #burn
        oj.algorithm.Algorithm_SingleSpinFlip_run(ising_model, [(1/t, N_sample)], callback_f)
    
    tmp = N_sample * len(np_H)
    tmp2 = N_sample * len(np_H)**2
    E /= tmp
    M /= tmp#(N_vec, *np.shape(temp))
    abs_M /= tmp
    E2 /= tmp2
    M2 /= tmp2
    dEdT = calculus_d1(E, temp)
    smooth_dEdT = (dEdT + np.roll(dEdT, -1) + np.roll(dEdT, 1))/3
    Tc = temp[np.argmax(smooth_dEdT)]

    return {'E':E, 'E2':E2, 'M':M, 'abs_M':abs_M, 'M2':M2, 'Tc': Tc}

