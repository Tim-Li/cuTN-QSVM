import time
import numpy as np
import pandas as pd
from itertools import combinations, chain, product
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_digits, fetch_openml
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from cuquantum import *
import cupy as cp
from cupy.cuda import nccl
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

# mpi setup
root = 0
comm_mpi = MPI.COMM_WORLD
rank, size = comm_mpi.Get_rank(), comm_mpi.Get_size()
device_id = rank % getDeviceCount()
cp.cuda.Device(device_id).use()
name = MPI.Get_processor_name()
print("MPI rank %d / %d on %s." % (rank, size, name))

# input data
mnist = fetch_openml('mnist_784')
X = mnist.data.to_numpy()
Y = mnist.target.to_numpy().astype(int)
class_list = [7,9]
c01 = np.where((Y == class_list[0])|(Y == class_list[1]))
X,Y = X[c01],Y[c01]
MAX=1600
data_train, label_train = X[:MAX],Y[:MAX]
X_train, X_val, Y_train, Y_val = train_test_split(data_train, label_train, test_size = 0.2, random_state=255)

if rank == root:
    print(f' qubits, [num train data, num list, num parti-list, num gpu], [exp_t, operand_t, path_t, contact_t, total_t]')

def data_prepare(n_dim, sample_train, sample_test, nb1, nb2):
    std_scale = StandardScaler().fit(sample_train)
    data = std_scale.transform(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)
    pca = PCA(n_components=n_dim, svd_solver="auto").fit(data)
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
def build_qsvm_qc(bsp_qc,n_dim,y_t,x_t):
    qc_1 = bsp_qc.assign_parameters(y_t).to_gate()
    qc_2 = bsp_qc.assign_parameters(x_t).inverse().to_gate()
    kernel_qc = QuantumCircuit(n_dim)
    kernel_qc.append(qc_1,list(range(n_dim)))
    kernel_qc.append(qc_2,list(range(n_dim)))
    return kernel_qc
def renew_operand(n_dim,oper_tmp,y_t,x_t):
    oper = oper_tmp.copy()
    n_zg, n_zy_g = [], []
    for d1 in y_t:
        z_g  = np.array([[np.exp(-1j*0.5*d1),0],[0,np.exp(1j*0.5*d1)]])
        n_zg.append(z_g)
        y_g  = np.array([[np.cos(d1/2),-np.sin(d1/2)],[np.sin(d1/2),np.cos(d1/2)]])
        n_zy_g.append(z_g)
        n_zy_g.append(y_g)
    oper[n_dim*2:n_dim*4] = cp.array(n_zy_g)
    oper[n_dim*5-1:n_dim*6-1] = cp.array(n_zg)
    n_zgd, n_zy_gd = [], []
    for d2 in x_t[::-1]:       
        z_gd  = np.array([[np.exp(1j*0.5*d2),0],[0,np.exp(-1j*0.5*d2)]])
        n_zgd.append(z_gd)  
        y_gd  = np.array([[np.cos(d2/2),np.sin(d2/2)],[-np.sin(d2/2),np.cos(d2/2)]])
        n_zy_gd.append(y_gd)
        n_zy_gd.append(z_gd)
    oper[n_dim*6-1:n_dim*7-1] = cp.array(n_zgd)
    oper[n_dim*8-2:n_dim*10-2] = cp.array(n_zy_gd)
    return oper
def data_partition(indices_list,size,rank):
    num_data = len(indices_list)
    chunk, extra = num_data // size, num_data % size
    data_begin = rank * chunk + min(rank, extra)
    data_end = num_data if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    data_index = range(data_begin,data_end)
    indices_list_rank = indices_list[data_begin:data_end]
    return indices_list_rank
def data_to_operand(n_dim,operand_tmp,data1,data2,indices_list):
    operand_list = []
    for i1, i2 in indices_list:
        n_op = renew_operand(n_dim,operand_tmp,data1[i1-1],data2[i2-1])
        operand_list.append(n_op) 
    return operand_list
def operand_to_amp(opers, network):
    amp_tmp = []
    with network as tn:
        for i in range(len(opers)):
            tn.reset_operands(*opers[i])     
            amp_tn = abs(tn.contract()) ** 2
            amp_tmp.append(amp_tn)
    return amp_tmp
def get_kernel_matrix(data1, data2, amp_data, indices_list, mode=None):
    amp_m = list(chain.from_iterable(amp_data))
    # print(len(amp),len(indices_list))
    kernel_matrix = np.zeros((len(data1),len(data2)))
    i = -1
    for i1, i2 in indices_list:
        i += 1
        kernel_matrix[i1-1][i2-1] = np.round(amp_m[i],8)
    if mode == 'train':
        kernel_matrix = kernel_matrix + kernel_matrix.T+np.diag(np.ones((len(data2))))
    return kernel_matrix

def run_tnsm(data_train, n_dim):
    #1. data partition
    list_train = list(combinations(range(1, len(data_train) + 1), 2))
    list_train_partition = data_partition(list_train,size,rank)
    
    #2. data to operand
    #2-1. quantum circuit setup and get exp
    t0 = time.time()   
    bsp_qc = make_bsp(n_dim)
    circuit = build_qsvm_qc(bsp_qc,n_dim, data_train[0], data_train[0])
    converter = CircuitToEinsum(circuit, dtype='complex128', backend='cupy')
    a = str(0).zfill(n_dim)
    exp, oper = converter.amplitude(a)     
    exp_t = round((time.time()-t0),3)

    #2-2. all data to operand
    t0 = time.time() 
    oper_train = data_to_operand(n_dim,oper,data_train,data_train,list_train_partition)    
    oper_t = round((time.time()-t0),3)

    #3. operand to amplitude
    #3-1. tensor network setup
    t0 = time.time()     
    options = NetworkOptions(blocking="auto",device_id=device_id)
    network = Network(exp, *oper,options=options)
    path, info = network.contract_path()     
    network.autotune(iterations=20)
    path_t = round((time.time()-t0),3)

    #3-2. all operand to amplitude
    t0 = time.time()
    oper_data = oper_train
    amp_list = operand_to_amp(oper_data, network)
    amp_train = cp.array(amp_list[:len(oper_train)])
    amp_data_train = comm_mpi.gather(amp_train, root=0)
    tnsm_kernel_t = round((time.time()-t0),3)

    if rank == root:
        print(f' {n_dim}, {len(data_train)}, {len(list_train)}, {len(list_train_partition)}, {len(amp_data_train)},  {exp_t}, {oper_t}, {path_t}, {tnsm_kernel_t}, {round((exp_t+oper_t+path_t+tnsm_kernel_t),3)}')

dd = np.zeros((20,2))
run_tnsm(dd, 2)
## for 1 node 8 gpus 
for ndim in [2,4,8,16,32,64,128,256,512,784]:
    for d in [20,40,50,60,80,100,200,400,500,600,800,1000]:
        dd = np.zeros((d,ndim))
        run_tnsm(dd, ndim)
## for 4 node 8 gpus 
# for ndim in [1024,2048,2352,3072,4096]:
#     for d in [20,40,50,60,80,100,200,400,500,600,800,1000]:
#         dd = np.zeros((d,ndim))
#         run_tnsm(dd, ndim)