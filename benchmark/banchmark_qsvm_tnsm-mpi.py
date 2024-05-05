import numpy as np
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
from cupy.cuda import nccl
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

root = 0
comm_mpi = MPI.COMM_WORLD
rank, size = comm_mpi.Get_rank(), comm_mpi.Get_size()
device_id = rank % getDeviceCount()
# Note that all NCCL operations must be performed in the correct device context.
cp.cuda.Device(device_id).use()

# Set up the NCCL communicator.
nccl_id = nccl.get_unique_id() if rank == root else None
nccl_id = comm_mpi.bcast(nccl_id, root)
comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)

mnist = fetch_openml('mnist_784')
X = mnist.data.to_numpy()
Y = mnist.target.to_numpy().astype(int)
class_list = [7,9]
c01 = np.where((Y == class_list[0])|(Y == class_list[1]))
X,Y = X[c01],Y[c01]
data_train, label_train = X[:2000],Y[:2000]
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
def new_op(n_dim,oper,y_t,x_t):
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
def kernel_matrix_tnsm(y_t, x_t, opers, indices_list, network, mode=None):
    kernel_matrix = np.zeros((len(y_t),len(x_t)))
    i = -1
    with network as tn:
        for i1, i2 in indices_list:
            i += 1
            tn.reset_operands(*opers[i])     
            amp_tn = abs(tn.contract()) ** 2
            kernel_matrix[i1-1][i2-1] = np.round(amp_tn,8) 
    if mode == 'train':
        kernel_matrix = kernel_matrix + kernel_matrix.T+np.diag(np.ones((len(x_t))))
    return kernel_matrix
def operand_to_amp(opers, indices_list, network):
    amp_tmp = []
    i = -1
    with network as tn:
        for i1, i2 in indices_list:
            i += 1
            tn.reset_operands(*opers[i])     
            amp_tn = abs(tn.contract()) ** 2
            amp_tmp.append(amp_tn)
    return amp_tmp


def run_tnsm(n_dim, nb1, nb2):
    data_train, data_val  = data_prepare(n_dim, X_train, X_val, nb1, nb2)
    bsp_qc = make_bsp(n_dim)
    bsp_kernel_tnsm = QuantumKernel(feature_map=bsp_qc)   
    indices_list_t = list(combinations(range(1, len(data_train) + 1), 2))

    t0 = time.time()      
    circuit = bsp_kernel_tnsm.construct_circuit(data_train[0], data_train[0],False)
    converter = CircuitToEinsum(circuit, dtype='complex128', backend='cupy')
    a = str(0).zfill(n_dim)
    exp, oper = converter.amplitude(a)     
    exp_t = round((time.time()-t0),3)

    num_data = len(indices_list_t)
    chunk, extra = num_data // size, num_data % size
    data_begin = rank * chunk + min(rank, extra)
    data_end = num_data if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    data_index = range(data_begin,data_end)
    indices_list_rank = indices_list_t[data_begin:data_end]
    # print(f"Process {rank} is processing data range: {data_index}.",num_data,len(indices_list_rank))
    
    t0 = time.time() 
    oper_train = []
    for i1, i2 in indices_list_rank:
        n_op = new_op(n_dim,oper,data_train[i1-1],data_train[i2-1])
        oper_train.append(n_op)        
    oper_t = round((time.time()-t0),3)

    t0 = time.time()     
    oper = oper_train[0]
    options = NetworkOptions(blocking="auto",device_id=device_id)
    network = Network(exp, *oper,options=options)
    path, info = network.contract_path()     
    network.autotune(iterations=20)
    path_t = round((time.time()-t0),3)
    
    t0 = time.time()     
    amp_list = operand_to_amp(oper_train, indices_list_rank, network)
    amp_list = cp.array(amp_list)

    # t0 = time.time() 
    stream_ptr = cp.cuda.get_current_stream().ptr
    # comm_nccl.reduce(amp_list.ptr, amp_list.ptr, amp_list.size, nccl.NCCL_FLOAT64, nccl.NCCL_SUM, root, stream_ptr)
    data = comm_mpi.gather(amp_list, root=0)
    # data = comm_nccl.allGather(amp_list, amp_list, amp_list.size, nccl.NCCL_FLOAT64, stream_ptr)
    tnsm_kernel_t = round((time.time()-t0),3)
    if rank == root:
        print(n_dim,exp_t,oper_t,path_t,tnsm_kernel_t,len(amp_list),len(data))

run_tnsm(2,40,1)
# run_tnsm(128,1000,1)
for d in [40,60,80,100,200,400,600,800,1000]:
    run_tnsm(128,d,1)
# run_tnsm(128,500,1)
# for q in range(2,34):
#     run_tnsm(q,2,1)
# for q in range(42,200,10):
#     run_tnsm(q,2,1)
# for q in [200,300,400,500,600,784]:
#     run_tnsm(q,2,1)