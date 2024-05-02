## cuTN-QSVM: cuTensorNet-accelerated Quantum Support Vector Machine with cuQuantum SDK

### Todo
1. build up qsvm simulation code by cuTensorNet
2. compare classification result with qiskit QSVC API
3. Multiprocessing for qiskit-to-TN transformation
4. Quantum Circuit with topology same as Google's and IBM's QC
5. CPU benchmark (compared with Numpy opt_einsum.contract)

### Install
```
pip install qiskit[visualization]==0.44.2
pip install qiskit-machine-learning==0.6.1
pip install -v --no-cache-dir cuquantum cuquantum-python
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install opt-einsum
```
### quick check env
```
python env_check.py
```

### Dataset
- fashion-mnist

### Simulation method
- **cuTensorNet** 

---
### Check
- Target：acc cutensornet simulation for QSVM
1. compare `qiskit machine learning` and `cutensornet simulation` on same datasets
2. compare QSVM simulation with `opt-einsum`, `opt-einsum-gpu` and `cutensornet`
3. banchmark `cutensornet` (single-gpu and multi-gpu) on A100
4. datasets
5. opt `cutensornet parameters` and `quantum circuit topology`
6. **create acc toolchain**
---
- check accuracy using cuTensorNetwork
    1. [check qiskit-qsvm code：add new feature](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html)
    2. select small scale datasets to check accuracy compare `our code[cuTensorNetwork]` and `qiskit-qsvm[cuStateVec]`
    3. [add opt_einsum code](https://optimized-einsum.readthedocs.io/en/stable/)
- check acc toolchain**
    1. [check how to create qsvm qc](https://qiskit-community.github.io/qiskit-machine-learning/apidocs/qiskit_machine_learning.kernels.html#module-qiskit_machine_learning.kernels)
    2. [check how cuquantum.CircuitToEinsum work](https://docs.nvidia.com/cuda/cuquantum/latest/python/api/generated/cuquantum.CircuitToEinsum.html)
    3. [Running with Threadpool and DASK](https://qiskit.org/ecosystem/aer/howtos/parallel.html)
    4. [Dask-MPI with GPUs](https://mpi.dask.org/en/latest/gpu.html)
---
