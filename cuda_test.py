# source: https://towardsdatascience.com/cuda-by-numba-examples-1-4-e0d06651612f

import numpy as np
import numba
from numba import cuda

print(np.__version__)
print(numba.__version__)

cuda.detect()


# Example 1.1: Add scalars
@cuda.jit
def add_scalars(a, b, c):
    c[0] = a + b


dev_c = cuda.device_array((1,), np.float32)

add_scalars[1, 1](2.0, 7.0, dev_c)

c = dev_c.copy_to_host()
print(f"2.0 + 7.0 = {c[0]}")
#  2.0 + 7.0 = 9.0
