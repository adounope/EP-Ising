import numpy as np
from datetime import datetime
import os
import src.EP_model as EP_model
from tqdm import tqdm
import src.utils as utils
import matplotlib.pyplot as plt
np.random.seed(80805)
x_train, y_train, _, x_val, y_val, _ = utils.load_data('./data/MNIST_1000')
print(x_train.shape)
print(x_val.shape)
x_train = x_train.reshape(len(x_train), -1)
x_val = x_val.reshape(len(x_val), -1)

########################################################################3
N_epoch = 200
input_size=196
hidden_size=64
output_size=40
N_spins = hidden_size + output_size
topology = np.ones((N_spins, N_spins), dtype=bool)
np.fill_diagonal(topology, False)
topology[:hidden_size, :hidden_size] = False   # hidden never directly connect with other hidden
topology[-output_size:, -output_size:] = False # output never directly connect with other output

ep_model = EP_model.EP_model(input_size, hidden_size, output_size,\
                             N_spins,\
                             J_connected=topology,\
                             N_neal_temps=(2**8, 2**8),\
                             N_neal_steps=(1, 1),\
                             neal_start_temp=(2**6, 2**6),\
                             output_duplicate=4,\
                             lr = 32/256, beta = 32)
###########################################################################

time_string = datetime.now().isoformat(timespec='minutes')
folder_name = f'./results/{ep_model.N_spins}_{time_string}_MNIST_1000_b{ep_model.beta}'
os.system(f'mkdir {folder_name}')
os.system(f'mkdir {folder_name}/weights')
os.system(f'touch {folder_name}/train_log.txt')
utils.save_obj(f'{folder_name}/ep_model.pkl', ep_model) # pickle cannot save openjij schedule object

ep_model.create_schedule()
with open(f'{folder_name}/train_log.txt', 'a') as f:
    f.write(f'beta={ep_model.beta}\tlr=1/{1/ep_model.lr}\tbatch_size = {ep_model.max_process}\tnum_walk = 2**({np.log2(ep_model.N_neal_temps)} + {np.log2(ep_model.N_neal_steps)})\trev_neal_temp = {ep_model.rev_neal_start_temp:.1f}\n')
with open(f'{folder_name}/acc_log.csv', 'a') as f:
    f.write(f'epoch, train_acc, val_acc\n')

train_batch_idx = utils.batch_idx(N=len(x_train), batch_size=ep_model.max_process)
val_batch_idx = utils.batch_idx(N=len(x_val), batch_size=ep_model.max_process)


best_val_acc = 0
acc_history = []
for epoch in range(N_epoch):
    N_train_correct = N_val_correct = 0
    for s, e in tqdm(train_batch_idx, desc=f'epoch {epoch}'): #start and end index of minibatch
        pred = ep_model.train(x_train[s:e], y_train[s:e])
        N_train_correct += (pred == ep_model.label_to_num(y_train[s:e])).sum()
    train_acc = N_train_correct / len(x_train)
    for s, e in val_batch_idx:
        pred, cfg = ep_model.forward(x_val[s:e])
        N_val_correct += (pred == ep_model.label_to_num(y_val[s:e])).sum()
    val_acc = N_val_correct / len(x_val)

    print(f'\ttrain_acc: {train_acc*100:.2f}%\tval_acc: {val_acc*100:.2f}%')

    acc_history.append([train_acc, val_acc])
    current_best = False
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        np.save(f'{folder_name}/weights/{epoch}_J', ep_model.J)
        np.save(f'{folder_name}/weights/{epoch}_H', ep_model.H)
        np.save(f'{folder_name}/weights/{epoch}_W', ep_model.W)
        current_best = True
    with open(f'{folder_name}/train_log.txt', 'a') as f:
        s = datetime.now().isoformat(timespec='seconds')
        f.write(f'{s}\n')
        f.write(f'\tepoch: {epoch}\ttrain_acc: {train_acc*100:.2f}%\tval_acc: {val_acc*100:.2f}%')
        if current_best:
            f.write('\tcurrent_best\n')
        else:
            f.write('\n')
    with open(f'{folder_name}/acc_log.csv', 'a') as f:
        f.write(f'{epoch}, {train_acc*100:.2f}, {val_acc*100:.2f}\n')


acc_history = np.array(acc_history)
np.save(f'{folder_name}/acc_history', acc_history)
plt.plot(np.arange(len(acc_history)), acc_history[:, 0]*100, label='train')
plt.plot(np.arange(len(acc_history)), acc_history[:, 1]*100, label='val')
plt.legend()
plt.ylabel('accuracy %')
plt.xlabel('epoch')
plt.title('acc history')
plt.ylim(0, 100)
plt.savefig(f'{folder_name}/acc_history.png')
        