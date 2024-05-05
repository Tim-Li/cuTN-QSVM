import numpy as np
import cupy as cp
import pandas as pd
import time
import matplotlib.pyplot as plt
from itertools import combinations,product
from multiprocessing import Pool
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import GridSearchCV
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.circuit import ParameterVector
from cuquantum import *
import time
import cupy as cp
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

root = 0
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
device_id = 0
cp.cuda.Device(device_id).use()

mnist = fetch_openml('mnist_784')
X = mnist.data.to_numpy()
Y = mnist.target.to_numpy().astype(int)
class_list = [7,9]
c01 = np.where((Y == class_list[0])|(Y == class_list[1]))
X,Y = X[c01],Y[c01]
data_train, label_train = X[:1000],Y[:1000]
X_train, X_val, Y_train, Y_val = train_test_split(data_train, label_train, test_size = 0.2, random_state=255)

def data_prepare(n_dim, sample_train, sample_test, nb1, nb2):
    std_scale = StandardScaler().fit(sample_train)
    data = std_scale.transform(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)
    pca = PCA(n_components=n_dim, svd_solver="full").fit(data)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)[:nb1]
    sample_test = minmax_scale.transform(sample_test)[:nb2]
    return sample_train, sample_test
def make_bsp(n_dim):
    param = ParameterVector("p",n_dim)
    bsp_qc = QuantumCircuit(n_dim)
    bsp_qc.h(list(range(n_dim)))
    i = 0
    for q in range(n_dim):
        bsp_qc.rz(param.params[q],[q])
        bsp_qc.ry(param.params[q],[q])
    for q in range(n_dim-1):
        bsp_qc.cx(0+i, 1+i)
        i+=1
    for q in range(n_dim):
        bsp_qc.rz(param.params[q],[q])
    return bsp_qc
def all_circuits_parallel(y_t, x_t, indices_list, n_dim, kernel, num_cpu):
    with Pool(processes=num_cpu, maxtasksperchild=100) as pool:
        circuits = pool.starmap(kernel.construct_circuit, [(y_t[i1-1], x_t[i2-1],False) for i1, i2 in indices_list])
    return circuits
def get_exp(x_t, n_dim, kernel):
    circuit = kernel.construct_circuit(x_t[0],x_t[0],measurement=False)
    converter = CircuitToEinsum(circuit, backend='numpy')
    a = str(0).zfill(n_dim)
    exp, oper = converter.amplitude(a)  
    return exp
def get_operand(circuit,n_dim):
    a = str(0).zfill(n_dim)
    converter = CircuitToEinsum(circuit, backend='numpy')
    oper = converter.amplitude_simple(a)  
    return oper    
def all_operands_parallel(circuit, n_dim, num_cpu):
    with Pool(processes=num_cpu, maxtasksperchild=100) as pool:
        indices_list = list(range(len(circuit)))
        operands = pool.starmap(get_operand, [(circuit[i],n_dim) for i in indices_list])
    return operands
def kernel_matrix_tnsm(y_t, x_t, exp, opers, indices_list, options, mode=None):
    kernel_matrix = np.zeros((len(y_t),len(x_t)))
    i, oper = -1, opers[0]
    with Network(exp, *oper, options=options) as tn:
        path, info = tn.contract_path()
        for i1, i2 in indices_list:
            i += 1
            tn.reset_operands(*opers[i])     
            amp_tn = abs(tn.contract()) ** 2
            kernel_matrix[i1-1][i2-1] = np.round(amp_tn,8) 
        tn.free()
    if mode == 'train':
        kernel_matrix = kernel_matrix + kernel_matrix.T+np.diag(np.ones((len(x_t))))
    return kernel_matrix

def run_tnsm(n_dim, nb1, nb2):
    data_train, data_val  = data_prepare(n_dim, X_train, X_val, nb1, nb2)
    bsp_qc = make_bsp(n_dim)
    bsp_kernel_tnsm = QuantumKernel(feature_map=bsp_qc)
    indices_list_t = list(combinations(range(1, len(data_train) + 1), 2))

    
    t0 = time.time()      
    circuit_train = all_circuits_parallel(data_train, data_train, indices_list_t, n_dim, bsp_kernel_tnsm, 10)
    circuit_t = round((time.time()-t0),3)
    t0 = time.time()        
    exp = get_exp(data_train, n_dim, bsp_kernel_tnsm)
    exp_t = round((time.time()-t0),3)
    t0 = time.time() 
    oper_train = all_operands_parallel(circuit_train, n_dim, 10)
    oper_t = round((time.time()-t0),3)
    t0 = time.time()     
    oper = oper_train[0]
    options = NetworkOptions(blocking="auto",device_id=device_id)
    path_t = round((time.time()-t0),3)
    t0 = time.time()     
    tnsm_kernel_matrix_train = kernel_matrix_tnsm(data_train, data_train, exp, oper_train, indices_list_t, options, mode='train')
    tnsm_kernel_t = round((time.time()-t0),3)
    print(n_dim,circuit_t,exp_t,oper_t,path_t,tnsm_kernel_t,len(circuit_train))

run_tnsm(2,2,1)
for q in range(2,100,10):
    run_tnsm(q,100,20)
# for q in range(102,200,10):
#     run_tnsm(q,100,20)
# for q in [200,300,400,500,600,784]:
#     run_tnsm(q,100,20)