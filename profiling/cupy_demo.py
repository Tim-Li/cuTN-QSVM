import cupy as cp
import nvtx
x = cp.arange(6).reshape(2, 3).astype('f')

def gpu_test():
    with nvtx.annotate("sum", color="blue"):
        x.sum(axis=1)

gpu_test()
gpu_test()
gpu_test()
gpu_test()
gpu_test()