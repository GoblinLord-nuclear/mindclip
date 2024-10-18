from mindspore import Tensor, ops
import numpy as np
n = np.ones(5)
t = Tensor(n)
np.add(n, 1, out=n)
print(f"n: {n}", type(n))
print(f"t: {t}", type(t))

