import numpy as np
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
import nvtx

traindata = pd.read_csv('../data/fashion-mnist-data/fashion-mnist_train.csv')
filt = (traindata['label'] < 2)
two_class = traindata.loc[filt]

def fashion_train_data(n_dim = 2,datasize = 2):
    with nvtx.annotate("load_data", color="blue"):
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

def qc_qiskit2cuquantum(x_t1, x_t2, n_dim):
    with nvtx.annotate("make_qc", color="green"):
        zz_map = ZZFeatureMap(feature_dimension=n_dim, reps=1, entanglement="linear", insert_barriers=True)
        zz_kernel = QuantumKernel(feature_map=zz_map)  
        zz_circuit = zz_kernel.construct_circuit(x_t1,x_t2,measurement=False,is_statevector_sim=False)
        converter = CircuitToEinsum(zz_circuit, backend='cupy')
        a = str(0).zfill(n_dim)
        exp, oper = converter.amplitude(a)  
    return zz_circuit, exp, oper

def get_path(exp, oper):
    with nvtx.annotate("find_path", color="blue"):
        qsvm_path, info = contract_path(exp,*oper)
    return qsvm_path

def get_kernel_matrix(x_t,datasize,n_dim,path):
    network_opts = NetworkOptions(blocking='auto')
    kernel_matrix = np.zeros((datasize,datasize))
    for i1 in range(2,datasize+1):
        for i2 in range(1,i1):
            # qc_tmp, exp_tmp, oper_tmp = qc_qiskit2cuquantum(x_t[i1-1],x_t[i2-1],n_dim)
            with nvtx.annotate("contract", color="red"):
                contract(exp,*oper, optimize = {'path' : path},options=network_opts) 
                # contract(exp,*oper, optimize = {'path' : path})     
    #         amp_tmp = np.round(np.sqrt(np.square(amp.real)+np.square(amp.imag)),5)
    #         kernel_matrix[i1-1][i2-1] = amp_tmp
    # kernel_matrix = kernel_matrix+kernel_matrix.T+np.diag(np.ones((datasize)))
    # return kernel_matrix  

n_dim, datasize  = 2, 100
x_t, y_t = fashion_train_data(n_dim, datasize)   
qc, exp, oper = qc_qiskit2cuquantum(x_t[0], x_t[0], n_dim)
path = get_path(exp, oper)
with nvtx.annotate("contract_total", color="blue"):
    get_kernel_matrix(x_t,datasize,n_dim,path)  


