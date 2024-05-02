pip install qiskit[visualization]==0.44.3
pip install qiskit-machine-learning==0.6.1
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install opt-einsum
conda install -y -c conda-forge cuquantum cuquantum-python cuda-version=12 #openmpi
#export LD_LIBRARY_PATH=/home/txm5780281/miniconda3/pkgs/libcublas-12.4.2.65-hac28a21_0/lib/:${LD_LIBRARY_PATH}
#export LD_LIBRARY_PATH=/home/txm5780281/miniconda3/pkgs/libnvjitlink-12.4.99-hac28a21_0/targets/sbsa-linux/lib:${LD_LIBRARY_PATH}
python qsvm/env_check.py
