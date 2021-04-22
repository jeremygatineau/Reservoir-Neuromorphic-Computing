import brian2 as br
import numpy as np
class Reservoir():
    def __init__(self, n_neurons_side=10):
        self.n_neurons_side=n_neurons_side
        self.lbd = 2
        self.M = np.zeros((n_neurons_side**3, n_neurons_side**3))
        idx = np.indices((n_neurons_side, n_neurons_side, n_neurons_side))
        self.indices = np.array([[[(y__, x__, z__) for x__, y__, z__ in zip(x_, y_, z_)] for x_, y_, z_ in zip(x, y, z)] for x,y, z in zip(idx[0], idx[1], idx[2])]).reshape([-1, 3])
        self.types = None
    def init_connection_matrix(self):
        self.types = ["ex" if np.random.rand() < 0.8 else "inh" for i in range(self.n_neurons_side**3)]
        C = np.array([[self._init_tw(i, j)*int(i!=j) for i in range(self.M.shape[1])] for j in range(self.M.shape[0])])
        self.M = np.array([[self._init_connection(i, j, C[i, j]) for i in range(self.M.shape[0])] for j in range(self.M.shape[0])])

    def _init_connection(self, i, j, c):
        posi = self.indices[i]
        posj = self.indices[j]
        d = np.linalg.norm(posi-posj)
        return 1 if np.random.rand() < c*np.exp(-d/self.lbd**2) else 0
    def _init_tw(self, i, j):
        if self.types[i] == "ex" and self.types[j]=="ex":
            return 0.3
        elif self.types[i] == "ex" and self.types[j] == "inh":
            return 0.2
        elif self.types[i] == "inh" and self.types[j] == "ex":
            return 0.4
        elif self.types[i] == "inh" and self.types[j] == "inh":
            return 0.1
        
    def encode_input(self):
        pass
    def read_out(self):
        pass
    def run(self):
        pass
    def _get_hash(self):
        pass
    