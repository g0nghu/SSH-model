'''
SSH model in OBC
Weixiang Qu
'''

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# The Hamiltonian of SSH model in OBC
def Ham_SSH_OBC(N,v,w):
    dd = []
    for i in range (2*N-1):
        if i%2 == 0:
            dd.append(v)
        else:
            dd.append(w)
    H = np.diag(dd,-1)+np.diag(dd,1)
    return H

## Calculate the energy and eigenstate
def Cal(N,vlist,w):
    Energies = np.empty([len(vlist),2*N])
    States = [None]*len(vlist)
    
    for i in range(len(vlist)):
        v = vlist[i]
        H = Ham_SSH_OBC(N,v,w)
        Energies[i,:],States[i] = LA.eigh(H)
    return Energies, States

if __name__ == "__main__":
    N = 12
    w = 1
    vlist = np.linspace(0,3,30)

    Energies,States = Cal(N,vlist,w)

    fig_energy = plt.figure(1)
    for i in range(2*N):
        plt.plot(vlist,Energies[:,i],color='black',linewidth=1)
    plt.xlabel("v")
    plt.ylabel("Energies")
    plt.annotate('wave:1', xy=(0.5, 0), xytext=(0.0, 2.5), arrowprops=dict(color='blue', headwidth=3, width=1))
    plt.annotate('wave:2', xy=(0.5, 0), xytext=(0.0, -2.5), arrowprops=dict(color='blue', headwidth=3, width=1))
    plt.annotate('wave:3', xy=(0.5, Energies[5,9]), xytext=(0.8, -2.5), arrowprops=dict(color='blue', headwidth=3, width=1))
    fig_energy.savefig("SSH_Energies.pdf")

    fig_wave1 = plt.figure(2)
    plt.bar(np.array(range(2*N))+1,States[5][:,12])
    plt.xlabel('site')
    plt.ylabel('wavefunction')
    plt.title("wave:1")
    fig_wave1.savefig("wave1.pdf")
    

    fig_wave2 = plt.figure(3)
    plt.bar(np.array(range(2*N))+1,States[5][:,11])
    plt.xlabel('site')
    plt.ylabel('wavefunction')
    plt.title("wave:2")
    fig_wave2.savefig("wave2.pdf")
    
    fig_wave3 = plt.figure(4)
    plt.bar(np.array(range(2*N))+1,States[5][:,9])
    plt.xlabel('site')
    plt.ylabel('wavefunction')
    plt.title("wave:3")
    fig_wave3.savefig("wave3.pdf")
    