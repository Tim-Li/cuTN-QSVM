import numpy as np
import cupy as cp
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
import qiskit
from cuquantum import *
import time
#import nvtx



traindata = pd.read_csv('../Data/fashion-mnist-data/fashion-mnist_train.csv')
filt = (traindata['label'] < 2)
two_class = traindata.loc[filt]



def fashion_train_data(n_dim = 2,datasize = 2):
    data_train  = two_class.iloc[:,1:785] / 255.0
    label_train = pd.DataFrame([two_class.iloc[:,0]]).T
    l_train=pd.DataFrame([two_class.iloc[:,0]]).T
    X_train, X_val, Y_train, Y_val = train_test_split(data_train, l_train, test_size = 0.25, random_state=255)
    X_train = StandardScaler().fit_transform(X_train)
    X_val   = StandardScaler().fit_transform(X_val)
    pca = PCA(n_components=n_dim).fit(X_train)
    X_train = pca.transform(X_train)
    X_train = X_train[:datasize]
    Y_train = np.array(Y_train).ravel()
    Y_train = Y_train[:datasize]  
    return X_train, Y_train



def get_circuit(x_t1, x_t2, n_dim):
    zz_map = ZZFeatureMap(feature_dimension=n_dim, reps=1, entanglement="linear", insert_barriers=True)
    zz_kernel = QuantumKernel(feature_map=zz_map)  
    zz_circuit = zz_kernel.construct_circuit(x_t1,x_t2,measurement=False,is_statevector_sim=False)
    return zz_circuit

def get_path(circuit):
    converter = CircuitToEinsum(circuit, backend='cupy')
    a = str(0).zfill(n_dim)
    exp, oper = converter.amplitude(a)  
    return exp, oper



import os
from itertools import combinations
from multiprocessing import Pool

print(os.cpu_count())
num_cpu = 30


def all_circuits(datasize, x_t, n_dim):
    t0 = time.time()
    
    circuits = []
    for i1, i2 in combinations(range(1, datasize + 1), 2):
        cir = get_circuit(x_t[i1-1], x_t[i2-1], n_dim)
        circuits.append(cir)
    print("Time =", time.time()-t0)
    return circuits


def all_circuits_parallel(datasize, x_t, n_dim):
    t0 = time.time()

    # Use Pool to parallelize the computation of circuits
    with Pool(processes=num_cpu-2, maxtasksperchild=100) as pool:
        indices_list = list(combinations(range(1, datasize + 1), 2))
        circuits = pool.starmap(get_circuit, [(x_t[i1-1], x_t[i2-1], n_dim) for i1, i2 in indices_list])
    print("Time =", time.time() - t0)
    return circuits


def all_operands(all_circuits):
    t0 = time.time()
    
    operands = []
    i = -1
    for i1 in range(2,datasize+1):
        for i2 in range(1,i1):
            i += 1
            _, oper = get_path(all_circuits[i])
            oper_gpu = [cp.asarray(tensor) for tensor in oper]
            operands.append(oper_gpu)
    print("Time =", time.time()-t0)
    
    return operands


def all_operands_parallel(all_circuits):
    t0 = time.time()

    # Use Pool to parallelize the computation of operands
    with Pool(processes=num_cpu-2, maxtasksperchild=100) as pool:
        results = pool.map(get_path, all_circuits)

    operands = [[cp.asarray(tensor) for tensor in oper] for _, oper in results]
    print("Time =", time.time() - t0)

    return operands



n_dim, datasize  = 2, 40
x_t, y_t = fashion_train_data(n_dim, datasize)



circuits = all_circuits(datasize, x_t, n_dim)
circuits = all_circuits_parallel(datasize, x_t, n_dim)
operands =  all_operands(circuits)
operands =  all_operands_parallel(circuits)



def get_kernel_matrix_gpu(operands,x_t,datasize,n_dim):
    kernel_matrix = cp.zeros((datasize,datasize))

    circuit = get_circuit(x_t[0], x_t[0], n_dim)
    exp, _ = get_path(circuit)
    oper, i = operands[0], -1
    with Network(exp, *oper) as tn:
        path, info = tn.contract_path()
        t0 = time.time()
        for i1 in range(2,datasize+1):
            for i2 in range(1,i1):
                i += 1
                tn.reset_operands(*operands[i])
                amp = tn.contract()
                kernel_matrix[i1-1][i2-1] = cp.round(cp.sqrt(cp.square(amp.real)+cp.square(amp.imag)),5)
        kernel_matrix = kernel_matrix + kernel_matrix.T+cp.diag(cp.ones((datasize)))
        print("Time = ", time.time()-t0)
    
    return kernel_matrix



""" TNs are on GPU, contract with cuquantum.Network specified (https://docs.nvidia.com/cuda/cuquantum/latest/python/api/generated/cuquantum.Network.html) """

get_kernel_matrix_gpu(operands, x_t, datasize, n_dim)