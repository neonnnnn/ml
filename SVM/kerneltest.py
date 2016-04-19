import kernel
import numpy as np
kernelname = "linear"
params = []

kernel = kernel.get_kernel(kernelname)(params)

print kernel.__class__.__name__
print kernel.calc_kernel(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[4, 5, 6], [7, 8, 9]]))[0]

