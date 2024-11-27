import numpy as np
import cupy as cp
import time

# Shape and batch size
batch_shape = (110, 8, 4)

# Generate random data
numpy_arrays = np.random.rand(*batch_shape)  # Generate data for NumPy
cupy_arrays = cp.random.rand(*batch_shape)   # Generate data directly on GPU for CuPy

# Warm up the GPU to initialize the CUDA context
_ = cp.array([1, 2, 3]).sum()
cp.cuda.Stream.null.synchronize()  # Ensure warm-up computation is complete

# Measure NumPy SVD time (loop over individual arrays)
start_numpy = time.time()
for i in range(batch_shape[0]):
    u, s, vh = np.linalg.svd(numpy_arrays[i], full_matrices=False)
end_numpy = time.time()
print(f"NumPy SVD time for {batch_shape[0]} arrays: {end_numpy - start_numpy:.6f} seconds")
while True:
    # Measure CuPy SVD time (batch computation)
    start_cupy = time.time()
    u_batch, s_batch, vh_batch = cp.linalg.svd(cupy_arrays, full_matrices=False)
    cp.cuda.Stream.null.synchronize()  # Ensure all GPU operations are complete
    end_cupy = time.time()
    print(f"CuPy SVD time for batch array: {end_cupy - start_cupy:.6f} seconds")
