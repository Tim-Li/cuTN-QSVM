[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

## cuTN-QSVM: cuTensorNet-accelerated Quantum Support Vector Machine with cuQuantum SDK
Welcome to the official repository of cuTN-QSVM, an advanced implementation of Quantum Support Vector Machines (QSVMs) facilitated by NVIDIA's cuQuantum SDK through the cuTensorNet library. This project epitomizes the integration of quantum computing technologies with state-of-the-art high-performance computing systems to elevate quantum machine learning to unprecedented levels of efficiency and scalability.

## Project Overview
Quantum Support Vector Machines offer a quantum-enhanced approach to solving complex, multidimensional classification problems, outstripping the capabilities of their classical counterparts under certain conditions. Despite their potential, the scalability of QSVMs is hampered by their exponential growth in computational requirements with increasing qubit counts. cuTN-QSVM leverages cuQuantum SDK's cuTensorNet library to mitigate this challenge, effectively reducing the computational complexity from exponential to polynomial time.

Technical Highlights:

- Efficient Quantum Simulations: Utilizing the cuTensorNet library, cuTN-QSVM drastically reduces the computational overhead of QSVMs, enabling the execution of quantum simulations for systems up to 784 qubits on the NVIDIA A100 GPU within mere seconds.
- Multi-GPU Processing: The implementation supports Multi-GPU processing via the Message Passing Interface (MPI), documenting significant reductions in computation times and demonstrating scalable performance improvements across increasing data sizes.
- Empirical Validation: In empirical assessments, cuTN-QSVM consistently achieves high classification accuracy, reaching up to 95% on the MNIST dataset for training sets exceeding 100 instances, thereby significantly surpassing the performance of traditional SVMs.


<a name="quickstart"></a>

## Quick Start 

### Installation
```
pip install qiskit[visualization]==0.44.2
pip install qiskit-machine-learning==0.6.1
pip install -v --no-cache-dir cuquantum cuquantum-python
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install opt-einsum
```
### Quick Environment Check
The env_check.py script is crafted to swiftly verify that your computational environment is optimally configured to execute simulations with cuTN-QSVM, leveraging the capabilities of cuQuantum and Qiskit. This Python script generates a random quantum circuit using Qiskit, then converts it to Einstein summation format utilizing cuQuantum's CircuitToEinsum with the CuPy backend. This process allows you to assess the integration and performance of these essential tools on your system. To run this script and ensure all necessary libraries are correctly interacting and prepared for more complex operations, execute the following command in your terminal:

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
