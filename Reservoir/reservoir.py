import brian2 as br
import numpy as np
class Reservoir():
    def __init__(self, n_neurons_side=10, lbd=2):
        self.n_neurons_side=n_neurons_side
        self.lbd = lbd
        self.M = np.zeros((n_neurons_side**3, n_neurons_side**3))
        idx = np.indices((n_neurons_side, n_neurons_side, n_neurons_side))
        self.indices = np.array([[[(x__, y__, z__) for x__, y__, z__ in zip(x_, y_, z_)] for x_, y_, z_ in zip(x, y, z)] for x,y, z in zip(idx[0], idx[1], idx[2])]).reshape([-1, 3])
        self.types = None
        self.input_group = None
        self.output_group = None
        self.reservoir_group = None
        self.SGen = None
        self.InSyn = None
        self.sII = None
        self.sIR = None
        self.sIO = None
        self.sRR = None
        self.sRO = None
        self.sOO = None
        self.network = br.Network()
        self.readOutMonitor = None

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
    
    def get_connection_matrices(self):
        n = self.n_neurons_side**3-2*self.n_neurons_side**2 # n_side**3 neurons minus the first and last faces of n_side**2
        II = np.zeros((self.n_neurons_side**2, self.n_neurons_side**2))
        IR = np.zeros((self.n_neurons_side**2, n))
        IO = np.zeros((self.n_neurons_side**2, self.n_neurons_side**2))
        RR = np.zeros((n, n))
        RO = np.zeros((n, self.n_neurons_side**2))
        OO = np.zeros((self.n_neurons_side**2, self.n_neurons_side**2))

        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                if self.indices[i][0] == 0: # first neuron in input group
                    if self.indices[j][0] == 0: # second neuron in input group
                        II[i, j] = self.M[i, j]
                    elif self.indices[j][0] == self.n_neurons_side-1: # second neuron in output group
                        IO[i, j-n-self.n_neurons_side**2] = self.M[i,j]
                    else: # second neuron in reservoir group
                        IR[i, j-self.n_neurons_side**2] = self.M[i,j]
                        
                elif self.indices[i][0] == self.n_neurons_side-1: # first neuron in output group
                    if self.indices[j][0] == self.n_neurons_side-1: # second neuron in output group
                        OO[i-n-self.n_neurons_side**2, j-n-self.n_neurons_side**2] = self.M[i,j]
                else : # first neuron in reservoir group
                    if self.indices[j][0] == self.n_neurons_side-1: # second neuron in output group
                        RO[i-self.n_neurons_side**2, j-n-self.n_neurons_side**2] = self.M[i,j]
                    elif self.indices[j][0] > 0:
                        RR[i-self.n_neurons_side**2, j-self.n_neurons_side**2] = self.M[i,j]
        return II, IR, IO, RR, RO, OO
    def set_neuron_groups(self):
        dyn_eqs = """
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I +j)/t2 : 1
        du/dt = 50*a*(b*v-u)/t2 : 1
        dj/dt = -j/tau : 1
        I : 1
        a : 1
        b : 1
        d : 1
        p : 1
        t2 : second
        tau : second
        """
        dyn_eqs_in = """
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I +j)/t2 : 1
        du/dt = 50*a*(b*v-u)/t2 : 1
        dj/dt = -j/tau : 1
        I = 1: 1
        a : 1
        b : 1
        d : 1
        p : 1
        t2 : second
        tau : second
        """
        reset_eqs = """
        v = -65
        u = u + d
        """

        self.input_group = br.NeuronGroup(self.n_neurons_side**2, dyn_eqs_in, threshold='v>30', reset=reset_eqs, method="euler")
        self.reservoir_group = br.NeuronGroup(self.n_neurons_side**3-2*self.n_neurons_side**2, dyn_eqs, threshold='v>30', reset=reset_eqs, method="euler")
        self.output_group = br.NeuronGroup(self.n_neurons_side**2, dyn_eqs, threshold='v>30', reset=reset_eqs, method="euler")
        
        n0 = self.n_neurons_side**2
        n1 = self.n_neurons_side**3-self.n_neurons_side**2
        n2 = self.n_neurons_side**3
        self.input_group.a = [0.02 if self.types[n]=='ex' else 0.1 for n in range(n0)]
        self.input_group.b = [0.2 for n in range(n0)]
        self.input_group.d = [8 if self.types[n]=='ex' else 2 for n in range(n0)]
        self.input_group.tau = [3*br.ms if self.types[n]=='ex' else 6*br.ms for n in range(n0)]
        self.input_group.t2 = [50*br.ms for n in range(n0)]
        self.input_group.p = [25 if self.types[n]=='ex' else -10 for n in range(n0)]

        self.reservoir_group.a = [0.02 if self.types[n]=='ex' else 0.1 for n in range(n0, n1)]
        self.reservoir_group.I = [np.random.rand()*5 if self.types[n]=='ex' else np.random.rand()*2 for n in range(n0, n1)]
        self.reservoir_group.b = [0.2 for n in range(n0, n1)]
        self.reservoir_group.d = [8 if self.types[n]=='ex' else 2 for n in range(n0, n1)]
        self.reservoir_group.tau = [3*br.ms if self.types[n]=='ex' else 6*br.ms for n in range(n0, n1)]
        self.reservoir_group.t2 = [50*br.ms for n in range(n0, n1)]
        self.reservoir_group.p = [25 if self.types[n]=='ex' else -10 for n in range(n0, n1)]

        self.output_group.a = [0.02 if self.types[n]=='ex' else 0.1 for n in range(n1, n2)]
        self.output_group.I = [np.random.rand()*5 if self.types[n]=='ex' else np.random.rand()*2 for n in range(n1, n2)]
        self.output_group.b = [0.2 for n in range(n1, n2)]
        self.output_group.d = [8 if self.types[n]=='ex' else 2 for n in range(n1, n2)]
        self.output_group.tau = [3*br.ms if self.types[n]=='ex' else 6*br.ms for n in range(n1, n2)]
        self.output_group.t2 = [50*br.ms for n in range(n1, n2)]
        self.output_group.p = [25 if self.types[n]=='ex' else -10 for n in range(n1, n2)]

    def connect_groups(self):
        CM = self.get_connection_matrices()
        ii, ir, io, rr, ro, oo = [m.nonzero() for m in CM] 
        self.sII = br.Synapses(self.input_group, self.input_group, on_pre='j_post += p_pre')
        self.sIR = br.Synapses(self.input_group, self.reservoir_group, on_pre='j_post += p_pre')
        self.sIO = br.Synapses(self.input_group, self.output_group, on_pre='j_post += p_pre')
        self.sRR = br.Synapses(self.reservoir_group, self.reservoir_group, on_pre='j_post += p_pre')
        self.sRO = br.Synapses(self.reservoir_group, self.output_group, on_pre='j_post += p_pre')
        self.sOO = br.Synapses(self.output_group, self.output_group, on_pre='j_post += p_pre')
    
        self.sII.connect(i=ii[0], j=ii[1])
        self.sIR.connect(i=ir[0], j=ir[1])
        self.sIO.connect(i=io[0], j=io[1])
        self.sRR.connect(i=rr[0], j=rr[1])
        self.sRO.connect(i=ro[0], j=ro[1])
        self.sOO.connect(i=oo[0], j=oo[1])

    def init_brian(self):
        self.set_neuron_groups()
        self.connect_groups()
        
        self.SGen = br.SpikeGeneratorGroup(self.n_neurons_side**2, range(self.n_neurons_side**2), [1*br.ms for _ in range(self.n_neurons_side**2)], period=50*br.ms)
        self.InSyn = br.Synapses(self.SGen, self.input_group, model='w : 1', on_pre='j_post += 100*w')
        self.InSyn.connect(j='i')

        self.readOutMonitor = br.SpikeMonitor(self.output_group, record=True)
        self.readInMonitor = br.SpikeMonitor(self.SGen, record=True)
        self.resSpikeMon = br.SpikeMonitor(self.reservoir_group, record=True)
        self.inSpikeMon = br.SpikeMonitor(self.input_group, record=True)
        self.stateInMonitor = br.StateMonitor(self.input_group, variables=['v', 'j', 'u'], record=True)
        self.stateResMonitor = br.StateMonitor(self.reservoir_group, variables=['v', 'j', 'u'], record=True)
        self.network.add(self.SGen, self.input_group, self.reservoir_group, self.output_group, \
                         self.InSyn, self.sII, self.sIO, self.sIR, self.sRR, self.sRO, \
                         self.sOO, self.readOutMonitor, self.readInMonitor, self.stateInMonitor, \
                         self.resSpikeMon, self.stateInMonitor, self.stateResMonitor)
        self.network.store(filename="reservoir_reset_state")
        
    def forward(self, image):
        self.network.restore(filename="reservoir_reset_state")
        assert image.max()<=1 and image.min()>=0, "image values must be between 0 and 1"
        img = 1-image 
        self.InSyn.w = img.reshape(-1)
        self.network.run(200*br.ms) # run enough so that signal has time to propagate
        return self.readOutMonitor.i, self.readOutMonitor.t/br.ms
    