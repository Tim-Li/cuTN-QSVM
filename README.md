[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)


## cuTN-QSVM: cuTensorNet-accelerated Quantum Support Vector Machine with cuQuantum SDK

<img align="right" width="200"
src="https://developer.nvidia.com/sites/default/files/akamai/nvidia-cuquantum-icon.svg"
/>

Welcome to the official repository of cuTN-QSVM, featuring fast GPU simulators for benchmarking Quantum Support Vector Machines (QSVMs) and scripts for generating compatible quantum circuits for hardware execution. Facilitated by NVIDIA's cuQuantum SDK and the cuTensorNet library, this project integrates cutting-edge quantum computing technologies with high-performance computing systems, enhancing quantum machine learning's efficiency and scalability to new heights.

## Project Overview
Quantum Support Vector Machines utilize a quantum-enhanced approach to tackle complex, multidimensional classification problems, surpassing the capabilities of classical SVMs under certain conditions. However, the scalability of QSVMs is traditionally limited due to the exponential growth in computational demands with increasing qubit counts. By employing NVIDIA's cuQuantum SDK and the cuTensorNet library, cuTN-QSVM effectively reduces this computational complexity from exponential to quadratic, enabling the simulation of large quantum systems up to 784 qubits on the NVIDIA A100 GPU within seconds.

Technical Highlights:

- Efficient Quantum Simulations:  The cuTensorNet library significantly lowers the computational overhead for QSVMs, facilitating rapid and efficient quantum simulations that can handle extensive qubit counts.
- Multi-GPU Processing: Supported by the Message Passing Interface (MPI), our implementation allows significant reductions in computation times and scalable performance improvements across varying data sizes.
- Empirical Validation: Through rigorous testing, cuTN-QSVM achieves high classification accuracy, with results reaching up to 95% on the MNIST dataset for training sets larger than 100 instances, markedly outperforming traditional SVMs.

<div style="text-align:center">
    <img src="figures/speedup_cutensornet.png" alt="Multi-GPU-Result" width="75%" height="auto">
</div>

<a name="quickstart"></a>

## Quick Start 

### Quick Environment Check
The env_check.py script is crafted to swiftly verify that your computational environment is optimally configured to execute simulations with cuTN-QSVM, leveraging the capabilities of cuQuantum and Qiskit. This Python script generates a random quantum circuit using Qiskit, then converts it to Einstein summation format utilizing cuQuantum's CircuitToEinsum with the CuPy backend. This process allows you to assess the integration and performance of these essential tools on your system. To run this script and ensure all necessary libraries are correctly interacting and prepared for more complex operations, execute the following command in your terminal:

```
python env_check.py
```

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

## Methodology
### cuTensorNet

NVIDIA's cuQuantum SDK includes cuTensorNet, a key component designed to optimize quantum circuit simulations on NVIDIA GPUs. It reduces computational costs and memory usage by streamlining tensor contractions and simplifying network complexities through its modular APIs. This enhancement enables efficient, large-scale simulations across multi-GPU and multi-node environments, advancing research in quantum physics, chemistry, and machine learning.

<div style="text-align:center">
    <img src="figures/cutensornet_module.png" alt="cuTensorNet Module" width="50%" height="auto">
</div>

### Workflow
In our enhanced QSVM simulation workflow using NVIDIA's cuQuantum SDK, the cuTensorNet module plays a pivotal role. This integration allows for the efficient transformation of quantum circuits into tensor networks, significantly reducing computational complexity from exponential to quadratic with respect to the number of qubits. By leveraging cuTensorNet’s advanced strategies like path reuse and non-blocking multi-GPU operations, we achieve substantial improvements in simulation speed and efficiency, enabling practical, large-scale quantum simulations up to 784 qubits.

<div style="text-align:center">
    <img src="figures/process_flow_comparison.png" alt="Process Flow" width="75%" height="auto">
</div>

### Multi-GPU Enable
In our study on distributed simulation within high-performance computing, we expanded QSVM model simulations using a multi-GPU setup to handle a dataset of over 1,000 MNIST images (28x28 pixels, 756 features). Leveraging NVIDIA’s cuStateVector with high-speed NVLink and MPI communication, we achieved significant computational efficiencies and demonstrated a linear speedup in quantum circuit simulations across multiple GPUs.

<div style="text-align:center">
    <img src="figures/multi_GPU_linearity.png" alt="Multi-GPU-Result" width="75%" height="auto">
</div>

## How to cite

If you used this package or framework for your research, please cite:

```text
@misc{2024cuTensorNetQSVM,
      title={cuTN-QSVM: cuTensorNet-accelerated Quantum Support Vector Machine with cuQuantum SDK}, 
      author={...........},
      year={2024},
      eprint={2404.13971},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
