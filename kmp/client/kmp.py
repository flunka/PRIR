import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
from array import array
from pycuda.compiler import SourceModule

import itertools

THREADS = 128
# x*4*2 <= GPU_MEMORY
# 1048576    == 1  MB
# 2097152    == 2  MB
# 4194304    == 4  MB
# 8388608    == 8  MB
# 16777216   == 16 MB
# 33554432   == 32 MB
# 67108864   == 64 MB
# 134217728  == 128MB max size on my gpu 
# 268435456  == 256MB out of memeory
# 536870912  == 512MB
# 1073741824 == 1  GB

MAX_SIZE = 134217728

mod = SourceModule("""
__global__ void KMP(unsigned int* pattern, unsigned int* target,int f[],int c[],int* n, int* m,int* result_counter)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int i = n[0] * index;
    int j = n[0] * (index + 2)-1;
    if(i>m[0])
        return;
    if(j>m[0])
        j=m[0];
    int k = 0;
    while (i < j)
    {
        if (k == -1)
        {
            i++;
            k = 0;

        }
        else if (target[i] == pattern[k])
        {
          if(i==1){
          }
          i++;
          k++;
          if (k == n[0])
          {

              c[i - n[0]] = i-n[0];
              atomicAdd(result_counter, 1);
              i = i - k + 1;
              k = 0;
          }
        }
        else{
          k = f[k];
        }

    }
    return;
}
""")


def build_partial_match_table(pattern):
  pm_table = np.zeros(len(pattern), dtype=np.int32)
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


def do_KMP(text, pattern, pm_table):
  start = cuda.Event()
  end = cuda.Event()
  KMP = mod.get_function("KMP")
  text = np.array(text)
  result = np.zeros_like(text, dtype=np.int32)
  result[:] = -1
  result_counter = np.array(0, dtype=np.int32)

  block = (THREADS, 1, 1)
  grid = (int((text.size / pattern.size + THREADS - 1) / THREADS), 1)

  n = pattern.size
  n = np.array(n, dtype=np.int32)

  m = text.size
  m = np.array(m, dtype=np.int32)

  start.record()
  KMP(cuda.In(pattern), cuda.In(text), cuda.In(pm_table), cuda.Out(result), cuda.In(n), cuda.In(m), cuda.InOut(result_counter), block=block, grid=grid)
  end.record()
  end.synchronize()

  # print("Time: {}ms".format(start.time_till(end)))

  return (result, result_counter.item(0))


def get_number_of_occurrence(pattern, filepath):

  text = array("u", "")
  pm_table = build_partial_match_table(pattern)
  part = 0

  pattern = list(pattern)
  pattern = np.array(pattern)

  result = 0

  with open(filepath) as f:
    for line in f:
      text += array("u", line)
      if(len(text) > MAX_SIZE):
        # print("Part {}:".format(part))
        result += do_KMP(text, pattern, pm_table)[1]
        text = array("u", "")
        part += 1

  if len(text) > 0:
    # print("Part {}:".format(part))
    result += do_KMP(text, pattern, pm_table)[1]

  return result


if __name__ == '__main__':
  pass
