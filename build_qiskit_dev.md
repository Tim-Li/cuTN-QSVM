```
cd /opt/nvidia
wget https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-23.10.0.6_cuda12-archive.tar.xz
tar -xvf cuquantum-linux-x86_64-23.10.0.6_cuda12-archive.tar.xz
ln -s /opt/nvidia/cuquantum-linux-x86_64-23.10.0.6_cuda12-archive /opt/nvidia/cuquantum 
rm cuquantum-linux-x86_64-23.10.0.6_cuda12-archive.tar.xz 
wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz 
tar -xvf libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz 
ln -s /opt/nvidia/libcutensor-linux-x86_64-1.7.0.1-archive /opt/nvidia/cutensor 
rm libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz
```
```
ln -s /opt/nvidia/cuquantum/lib/libcustatevec_static.a /usr/lib/libcustatevec_static.a
ln -s /opt/nvidia/cuquantum/lib/libcutensornet_static.a /usr/lib/libcutensornet_static.a
```
```
export CUQUANTUM_ROOT=/opt/nvidia/cuquantum
export CUTENSOR_ROOT=/opt/nvidia/cutensor
export LD_LIBRARY_PATH=$CUQUANTUM_ROOT/lib:$CUTENSOR_ROOT/lib/12:$LD_LIBRARY_PATH
```
```
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev -y
sudo apt-get install git build-essential libopenblas-dev -y
git clone https://github.com/Qiskit/qiskit-aer
cd qiskit-aer
pip install -r requirements-dev.txt && pip install pybind11 auditwheel
python ./setup.py bdist_wheel -- -DAER_MPI=True -DAER_DISABLE_GDR=True -DAER_THRUST_BACKEND=CUDA -DAER_CUDA_ARCH="7.0; 7.5; 8.0" -DCUQUANTUM_ROOT=/opt/nvidia/cuquantum -DCUTENSOR_ROOT=/opt/nvidia/cutensor -DAER_ENABLE_CUQUANTUM=true -DCUQUANTUM_STATIC=true --
pip install -U dist/qiskit_aer*.whl
pip install -v --no-cache-dir cuquantum cuquantum-python
```

<!-- apt install -y cmake && pip install pybind11 pluginbase patch-ng node-semver==0.6.1 bottle PyJWT fasteners distro colorama conan==1.59.0 scikit-build
apt-get install -y git wget -->