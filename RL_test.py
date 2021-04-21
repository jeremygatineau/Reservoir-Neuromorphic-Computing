import gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import progressbar
from IPython import display
from torch.autograd import Variable
from time import sleep
class model(nn.Module):
    def __init__(self, input_shape=8, n_layers=5, width=100, n_out=4):
        super(model, self).__init__()
        self.input_shape = input_shape

    
        self.layers = nn.ModuleList(
            [nn.Linear(input_shape, 64), nn.ReLU(), \
            nn.Linear(64, 128), nn.ReLU(), \
            nn.Linear(128, 256), nn.Sigmoid(), \
            nn.Linear(256, n_out)])

    
    
    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x

env=gym.make('LunarLander-v2').env


policy = model()
policy.load_state_dict(torch.load("model_700_sharon3_2", map_location=torch.device('cpu')))

done = 0
rs = []

plt.ion()
for _ in range(100):
    print(_)
    acc_r=0
    s = env.reset()

    while not done:
        st = torch.tensor([s], dtype=torch.float32)
        print(st)
        q_vals = policy(st)
        action = np.argmax(q_vals.cpu().detach().numpy())
        s_, r, done, _ = env.step(action)
        acc_r+=r
        s = np.copy(s_)
        
    rs.append(acc_r)
        


fig, axs = plt.subplots(1, 1, figsize=(20, 10))

axs.plot(rs)
axs.set_title('accumulated reward')
axs.set_xlabel('Epoch')
axs.set_ylim([0, 360])
plt.grid(True)
plt.show()
