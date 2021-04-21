import brian2 as br
import numpy as np
import matplotlib.pyplot as plt

br.start_scope()

input_eqs = '''
dv/dt = (I-v)/tau: 1
I : 1
tau : second
p : 1
'''
in_mat = np.zeros(5)
in_mat[0] = 1
input_group = br.NeuronGroup(len(in_mat), input_eqs, threshold='v>0.9', reset='v = 0', method='euler')
input_group.tau = 1*br.ms
input_group.I = in_mat
input_group.p = 0
N_res = 50  
eqs = '''
dv/dt = (I-v)/tau : 1 (unless refractory)
I = 0: 1
tau : second
'''
#+ 5*xi*tau**-0.5
res = br.NeuronGroup(N_res, eqs, threshold='v>1', reset='v = 0', refractory=1*br.ms, method='euler')
res.tau = 15*br.ms
S = br.Synapses(res, res, on_pre='v_post += 0.2')
S.connect(condition='i!=j', p=0.8)
IS = br.Synapses(input_group, res, on_pre='v_post += 0.4')
IS.connect(i=range(len(in_mat)), j=range(len(in_mat)))

M = br.SpikeMonitor(res)
N = br.StateMonitor(input_group, 'v', record=True)
p = 0
@br.network_operation(dt=50*br.ms)
def change_I():
    input_group.p +=1
    in_mat = np.zeros(5)
    print(input_group.p[0])
    p = int(input_group.p[0])
    input_group.I = 0
    in_mat[p] = 1
    input_group.I = in_mat
    

br.run(200*br.ms)


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    plt.plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')

visualise_connectivity(S)
print(M.t, M.i)
plt.figure(2, figsize=(10, 4))
plt.plot(M.t/br.ms, M.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')


plt.figure(3, figsize=(10, 4))
plt.plot(N.t/br.ms, N.v.T)
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.show()
