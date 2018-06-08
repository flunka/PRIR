import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

import itertools

THREADS = 1

mod = SourceModule("""
__global__ void KMP(unsigned int* pattern, unsigned int* target,int f[],int c[],int* n, int* m)
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


def get_number_of_occurrence(pattern, filepath):
  KMP = mod.get_function("KMP")
  with open(filepath) as f:
    text = f.read().splitlines()
  text = [list(line) for line in text]
  text = list(itertools.chain.from_iterable(text))
  text = np.array(text)

  pm_table = build_partial_match_table(pattern)

  pattern = list(pattern)
  pattern = np.array(pattern)

  result = np.zeros(text.size, dtype=np.int32)
  result[:] = -1

  block = (THREADS, 1, 1)
  grid = (int((text.size / pattern.size + THREADS - 1) / THREADS), 1)

  n = pattern.size
  n = np.array(n, dtype=np.int32)

  m = text.size
  m = np.array(m, dtype=np.int32)
  KMP(cuda.In(pattern), cuda.In(text), cuda.In(pm_table), cuda.InOut(result), cuda.In(n), cuda.In(m), block=block, grid=grid)

  return len(list(filter(lambda x: x >= 0, result)))


if __name__ == '__main__':
  pass
