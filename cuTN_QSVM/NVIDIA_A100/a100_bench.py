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
import os
from itertools import combinations,product
from multiprocessing import Pool
from sklearn.svm import SVC
import opt_einsum
print(os.cpu_count())

traindata = pd.read_csv('../Data/fashion-mnist-data/fashion-mnist_train.csv')

# data
def data_per_class(label,ds_train,ds_test):
    filt = (traindata['label'] == label)
    one_class_train = traindata.loc[filt][:ds_train]
    one_class_test  = traindata.loc[filt][ds_train:ds_train+ds_test]
    return one_class_train, one_class_test
def fashion_data(data0,data1,n_dim):
    full_class = pd.concat([data0,data1]).sample(frac = 1)
    data_full  = full_class.iloc[:,1:785] / 255.0
    data_full  = StandardScaler().fit_transform(data_full)
    pca        = PCA(n_components=n_dim).fit(data_full)
    data_full  = pca.transform(data_full)
    label_full = pd.DataFrame([full_class.iloc[:,0]]).T
    label_full = np.array(label_full).ravel()
    return data_full,label_full

# quantum circuit
def get_qc_circuit(x_t1, x_t2, n_dim):
    zz_map = ZZFeatureMap(feature_dimension=n_dim, reps=1, entanglement="linear", insert_barriers=True)
    zz_kernel = QuantumKernel(feature_map=zz_map)
    zz_circuit = zz_kernel.construct_circuit(x_t1,x_t2,measurement=False,is_statevector_sim=False)   
    return zz_circuit
def all_circuits(datasize, x_t, n_dim):
    circuits = []
    for i1, i2 in combinations(range(1, datasize + 1), 2):
        cir = get_qc_circuit(x_t[i1-1], x_t[i2-1], n_dim)
        circuits.append(cir)
    return circuits
def all_circuits_parallel(datasize, x_t, n_dim, num_cpu):
    # Use Pool to parallelize the computation of circuits
    with Pool(processes=num_cpu-2, maxtasksperchild=100) as pool:
        indices_list = list(combinations(range(1, datasize + 1), 2))
        circuits = pool.starmap(get_qc_circuit, [(x_t[i1-1], x_t[i2-1], n_dim) for i1, i2 in indices_list])
    all_circuits_parallel_t = time.time()-t0
    return circuits
def all_circuits_test_parallel(data1,data2, n_dim, num_cpu = 12):
    # Use Pool to parallelize the computation of circuits
    with Pool(processes=num_cpu-2, maxtasksperchild=100) as pool:
        indices_list = list(product(range(1, len(data1) + 1),range(1, len(data2) + 1)))
        circuits = pool.starmap(get_qc_circuit, [(data1[i1-1], data2[i2-1], n_dim) for i1, i2 in indices_list])
    return circuits

# operands
def get_oper(circuit):
    converter = CircuitToEinsum(circuit, backend='numpy')
    a = str(0).zfill(n_dim)
    exp, oper = converter.amplitude(a)
    return exp, oper
def all_operands(all_circuits,datasize):
    operands = []
    i = -1
    for i1 in range(2,datasize+1):
        for i2 in range(1,i1):
            i += 1
            _, oper = get_oper(all_circuits[i])
            oper_gpu = [cp.asarray(tensor) for tensor in oper]
            operands.append(oper_gpu)
    return operands
def all_operands_parallel(all_circuits,num_cpu):
    # Use Pool to parallelize the computation of operands
    with Pool(processes=num_cpu-2, maxtasksperchild=100) as pool:
        results = pool.map(get_oper, all_circuits)
    operands = [[cp.asarray(tensor) for tensor in oper] for _, oper in results]
    return operands

# kernel_matrix
def get_kernel_matrix_cutensornet(operands,x_t,datasize,n_dim):
    kernel_matrix = cp.zeros((datasize,datasize))
    circuit = get_qc_circuit(x_t[0], x_t[0], n_dim)
    exp, _ = get_oper(circuit)
    oper, i = operands[0], -1
    with Network(exp, *oper) as tn:
        path, info = tn.contract_path()
        for i1 in range(2,datasize+1):
            for i2 in range(1,i1):
                i += 1
                tn.reset_operands(*operands[i])
                amp = tn.contract()
                kernel_matrix[i1-1][i2-1] = cp.round(cp.sqrt(cp.square(amp.real)+cp.square(amp.imag)),5)
        kernel_matrix = kernel_matrix + kernel_matrix.T+cp.diag(cp.ones((datasize)))
    return kernel_matrix

def get_kernel_matrix_opteinsum(operands,x_t,datasize,n_dim):
    kernel_matrix = cp.zeros((datasize,datasize))
    circuit = get_qc_circuit(x_t[0], x_t[0], n_dim)
    exp, _ = get_oper(circuit)
    oper, i = operands[0], -1
    for i1 in range(2,datasize+1):
        for i2 in range(1,i1):
            i += 1
            amp = opt_einsum.contract(exp,*operands[i])
            kernel_matrix[i1-1][i2-1] = cp.round(cp.sqrt(cp.square(amp.real)+cp.square(amp.imag)),5)
    kernel_matrix = kernel_matrix + kernel_matrix.T+cp.diag(cp.ones((datasize)))
    return kernel_matrix

def get_kernel_matrix_cutensornet_n(operands,x_t,datasize1,datasize2,n_dim):
    kernel_matrix = cp.zeros((datasize1,datasize2))
    circuit = get_qc_circuit(x_t[0], x_t[0], n_dim)
    exp, _ = get_oper(circuit)
    oper, i = operands[0], -1
    with Network(exp, *oper) as tn:
        path, info = tn.contract_path()
        for i1 in range(1,datasize1+1):
            for i2 in range(1,datasize2+1):
                i += 1
                tn.reset_operands(*operands[i])
                amp = tn.contract()
                kernel_matrix[i1-1][i2-1] = cp.round(cp.sqrt(cp.square(amp.real)+cp.square(amp.imag)),5)
    return kernel_matrix

# start
## acc code
print('acc code')
num_cpu = os.cpu_count()
num_train = 50
num_test  = 20
class_lb = [0,1]
svc = SVC(kernel="precomputed")
data_0_train, data_0_test = data_per_class(class_lb[0],num_train,num_test)
data_1_train, data_1_test = data_per_class(class_lb[1],num_train,num_test)

runtime_data = pd.DataFrame([['qubits','score_train','score_test','data_t','circuits_t','operands_t','kernel_matrix_t_cu','kernel_matrix_t_op','total_t']])
runtime_data.to_csv("acc_code_t.csv", index=False,header=None,mode='a')
for q in range(2,20):
    n_dim=q
    tf = time.time()
    t0 = time.time()
    data_train, label_train = fashion_data(data_0_train,data_1_train,n_dim)
    data_test, label_test   = fashion_data(data_0_test,data_1_test,n_dim)
    data_t = time.time()-t0
    t0 = time.time()
    circuits_p = all_circuits_parallel(num_train*2,data_train,n_dim,num_cpu)
    circuits_test_p = all_circuits_test_parallel(data_test,data_train,n_dim, num_cpu)
    circuits_t = time.time()-t0  
    t0 = time.time()
    operands_p = all_operands_parallel(circuits_p,num_cpu)
    operands_test_p = all_operands_parallel(circuits_test_p,num_cpu)
    operands_t = time.time()-t0
    t0 = time.time()
    kernel_matrix_train_cu = get_kernel_matrix_cutensornet(operands_p,data_train,num_train*2,n_dim)
    kernel_matrix_t_cu = time.time()-t0
    kernel_matrix_test_cu  = get_kernel_matrix_cutensornet_n(operands_p, data_train, len(data_test),len(data_train), n_dim)
    # t0 = time.time()
    # kernel_matrix_train_op = get_kernel_matrix_opteinsum(operands_p,data_train,num_train*2,n_dim)
    # kernel_matrix_t_op = time.time()-t0
    svc.fit(kernel_matrix_train_cu.get(),label_train)
    score_cu_train = svc.score(kernel_matrix_train_cu.get(),label_train)
    score_cu_test  = svc.score(kernel_matrix_test_cu.get(),label_test)
    total_t = time.time()-tf
    print([n_dim,score_cu_train,score_cu_test,data_t,circuits_t,operands_t,kernel_matrix_t_cu,total_t])
    runtime_data = pd.DataFrame([[n_dim,score_cu_train,score_cu_test,data_t,circuits_t,operands_t,kernel_matrix_t_cu,total_t]])
    runtime_data.to_csv("acc_code_t.csv", index=False,header=None,mode='a')
print('done')

# start
## normal code
print('normal code')
num_cpu = None
num_train = 20
num_test  = 2
class_lb = [0,1]
svc = SVC(kernel="precomputed")
data_0_train, data_0_test = data_per_class(class_lb[0],num_train,num_test)
data_1_train, data_1_test = data_per_class(class_lb[1],num_train,num_test)

runtime_data = pd.DataFrame([['q','score','data_t','circuits_t','operands_t','kernel_matrix_t','total_t']])
runtime_data.to_csv("normal_code.csv", index=False,header=None,mode='a')
for q in range(2,29):
    n_dim=q
    tf = time.time()
    t0 = time.time()
    data_train, label_train = fashion_data(data_0_train,data_1_train,n_dim)
    data_t = time.time()-t0
    t0 = time.time()
    circuits = all_circuits(num_train*2,data_train,n_dim)
    circuits_t = time.time()-t0
    t0 = time.time()
    operands = all_operands(circuits,num_train*2)
    operands_t = time.time()-t0
    t0 = time.time()
    kernel_matrix_train_cu = get_kernel_matrix_cutensornet(operands,data_train,num_train*2,n_dim)
    kernel_matrix_t_cu = time.time()-t0
    t0 = time.time()
    kernel_matrix_train_op = get_kernel_matrix_opteinsum(operands,data_train,num_train*2,n_dim)
    kernel_matrix_t_op = time.time()-t0
    svc.fit(kernel_matrix_train_cu.get(),label_train)
    score_cu = svc.score(kernel_matrix_train_cu.get(),label_train)
    svc.fit(kernel_matrix_train_op.get(),label_train)
    score_op = svc.score(kernel_matrix_train_op.get(),label_train)
    total_t = time.time()-tf
    print([n_dim,score_cu,score_op,data_t,circuits_t,operands_t,kernel_matrix_t_cu,kernel_matrix_t_op,total_t])
    runtime_data = pd.DataFrame([[n_dim,score_cu,score_op,data_t,circuits_t,operands_t,kernel_matrix_t_cu,kernel_matrix_t_op,total_t]])
    runtime_data.to_csv("normal_code.csv", index=False,header=None,mode='a')
print('done')
