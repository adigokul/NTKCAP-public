import cupy as cp
print("Available GPU devices:", cp.cuda.runtime.getDeviceCount())