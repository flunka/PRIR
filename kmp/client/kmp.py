import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule


def build_partial_match_table(pattern):
  pm_table = np.zeros(len(pattern), dtype=int)
  pm_table[0] = -1
  for i in range(1, pm_table.size):
    k = pm_table[i - 1]
    while k >= 0:
      if pattern[k] == pattern[i - 1]:
        break
      else:
        k = pm_table[k]
    pm_table[i] = k + 1
  return pm_table


if __name__ == '__main__':
  pass
