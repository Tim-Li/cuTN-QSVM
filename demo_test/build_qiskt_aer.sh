sudo apt-get install git build-essential libopenblas-dev -y
git clone https://github.com/Qiskit/qiskit-aer
cd qiskit-aer
pip install -r requirements-dev.txt
pip install pybind11 auditwheel
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 cuquantum-cu12
# conda install -y -c conda-forge cuquantum cuquantum-python cuda-version=12
# python ./setup.py bdist_wheel -- -DAER_MPI=True -DAER_DISABLE_GDR=True
python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DCUQUANTUM_ROOT=/home/tim/anaconda3/envs/qsvm_v2/lib/python3.10/site-packages/cuquantum -DCUTENSOR_ROOT=/home/tim/anaconda3/envs/qsvm_v2/lib/python3.10/site-packages/cutensor -DAER_ENABLE_CUQUANTUM=true -DCUQUANTUM_STATIC=true --

python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DCUQUANTUM_ROOT=/home/tim/anaconda3/envs/qsvm_v2/lib/python3.10/site-packages/cuquantum -DCUTENSOR_ROOT=/home/tim/anaconda3/envs/qsvm_v2/lib/python3.10/site-packages/cutensor -DAER_ENABLE_CUQUANTUM=true -DCUQUANTUM_STATIC=true --


pip install -U dist/qiskit_aer*.whl

/home/tim/anaconda3/envs/qsvm_v2/lib/python3.10/site-packages/cuquantum/include/custatevec.h
/home/tim/anaconda3/envs/qsvm_v2/lib/python3.10/site-packages/cutensor/include/cutensor.h

/usr/bin/ld: cannot find -lcustatevec_static
/usr/bin/ld: cannot find -lcutensornet_static
/usr/bin/ld: cannot find -lcutensor_static

/home/tim/anaconda3/envs/qsvm_v2/lib/python3.10/site-packages/cuquantum/lib/libcustatevec.so.1
/home/tim/anaconda3/envs/qsvm_v2/lib/python3.10/site-packages/cuquantum/lib/libcutensornet.so.2

/home/tim/anaconda3/envs/qsvm_v2/lib/python3.10/site-packages/cutensor/lib/libcutensorMg.so.1
/home/tim/anaconda3/envs/qsvm_v2/lib/python3.10/site-packages/cutensor/lib/libcutensor.so.1

export CUTENSOR_ROOT=${PWD}/libcutensor
export LD_LIBRARY_PATH=${CUTENSOR_ROOT}/lib/12/:${LD_LIBRARY_PATH}


wget https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-23.03.0.20-archive.tar.xz
wget https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-23.10.0.6_cuda12-archive.tar.xz
tar -xvf cuquantum-linux-x86_64-23.03.0.20-archive.tar.xz 
ln -s cuquantum-linux-x86_64-23.03.0.20-archive cuquantum 
rm cuquantum-linux-x86_64-23.03.0.20-archive.tar.xz

wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz 
tar -xvf libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz 
ln -s libcutensor-linux-x86_64-1.7.0.1-archive cutensor 
rm libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz

export CUQUANTUM_ROOT=/opt/nvidia/cuquantum
export CUTENSOR_ROOT=/opt/nvidia/cutensor
export LD_LIBRARY_PATH=$CUQUANTUM_ROOT/lib/12:$CUTENSOR_ROOT/lib/12:$LD_LIBRARY_PATH

python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DAER_CUDA_ARCH="7.0; 7.5; 8.0" -DCUQUANTUM_ROOT=/opt/nvidia/cuquantum -DCUTENSOR_ROOT=/opt/nvidia/cutensor -DAER_ENABLE_CUQUANTUM=true -DCUQUANTUM_STATIC=true --