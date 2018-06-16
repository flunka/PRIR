import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
import sys
import argparse
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
# 134217728  == 128MB
# 268435456  == 256MB max size on my gpu
# 536870912  == 512MB out of memeory
# 1073741824 == 1  GB

MAX_SIZE = 268435456

mod = SourceModule("""
__global__ void KMP(unsigned char* pattern, unsigned char* target,int f[], unsigned char *c,int* n, int* m,int* result_counter)
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
        c[i*2+1]=target[i];
        if (k == -1)
        {
            i++;
            k = 0;

        }
        else if (target[i] == pattern[k])
        {          
          i++;
          k++;
          if (k == n[0])
          {
              atomicAdd(result_counter, 1);
              //c[atomicAdd(result_counter, 1)] = i-n[0];
              c[i*2]='*';
              c[(i-n[0])*2]='*';
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
  k = 0
  for i in range(1, pm_table.size):
    if (pattern[i] == pattern[k]):
      pm_table[i] = pm_table[k]
    else:
      pm_table[i] = k
      k = pm_table[k]
    while (k >= 0 and pattern[i] != pattern[k]):
      k = pm_table[k]
    k += 1
  return pm_table


def do_KMP(text, pattern, pm_table):
  start = cuda.Event()
  end = cuda.Event()
  KMP = mod.get_function("KMP")
  text = np.array(text)
  result = np.zeros(text.size * 2 + 1, dtype=np.uint8)
  result[:] = 35  # 35 == #
  result_counter = np.array(0, dtype=np.int32)

  block = (THREADS, 1, 1)
  grid = (int((text.size / pattern.size + THREADS - 1) / THREADS), 1)

  n = pattern.size
  n = np.array(n, dtype=np.int32)

  m = text.size
  m = np.array(m, dtype=np.int32)
  start.record()
  KMP(cuda.In(pattern), cuda.In(text), cuda.In(pm_table), cuda.InOut(result), cuda.In(n), cuda.In(m), cuda.InOut(result_counter), block=block, grid=grid)
  end.record()
  end.synchronize()

  # print("Time: {}ms".format(start.time_till(end)))

  return (result_counter.item(0), result)


def build_patterns(expression):
  patterns = [""]
  index = 0
  while (index < len(expression)):
    if (expression[index] == "["):
      index += 1
      old_patterns = patterns[:]
      first = True
      while (expression[index] != "]"):
        if (index + 1 >= len(expression) or expression[index].isalpha() == False):
          raise ValueError('Invalid regex!!!')
        if (first):
          first = False
          for i in range(0, len(patterns)):
            patterns[i] += expression[index]
        else:
          for i in range(0, len(old_patterns)):
            patterns.append(old_patterns[i] + expression[index])
        index += 1
      index += 1
    elif (expression[index] == "\\"):
      index += 1
      if (expression[index] == "d"):
        index += 1
        old_patterns = patterns[:]
        for x in range(0, 10):
          if (x == 0):
            for i in range(0, len(patterns)):
              patterns[i] += str(x)
          else:
            for i in range(0, len(old_patterns)):
              patterns.append(old_patterns[i] + str(x))
    else:
      for i in range(0, len(patterns)):
        patterns[i] += expression[index]
      index += 1
  for pattern in patterns:
    yield pattern


def get_number_of_occurrence(pattern, filepath):
  return find_pattern(pattern, filepath)


def find_pattern(pattern, filepath, print_result=False):
  text = array("B", "".encode())
  pm_table = build_partial_match_table(pattern)

  pattern = array("B", pattern.encode())
  pattern = np.array(pattern)

  number_of_occurrence = 0

  with open(filepath) as f, open("{}_result_{}.txt".format(filepath, pattern.tostring().decode('utf-8')), "w") as f2:
    for line in f:
      text += array("B", line.encode())
      if(len(text) > MAX_SIZE):
        result = do_KMP(text, pattern, pm_table)
        number_of_occurrence += result[0]
        if(print_result):
          f2.write(''.join(map(str, result[1][~(result[1] == 35)].tostring().decode('utf-8'))))
        text = array("B", "".encode())

    if len(text) > 0:
      result = do_KMP(text, pattern, pm_table)
      number_of_occurrence += result[0]
      if(print_result):
        f2.write(''.join(map(str, result[1][~(result[1] == 35)].tostring().decode('utf-8'))))

  return number_of_occurrence


def main():
  parser = argparse.ArgumentParser(description='KMP string matching algorithm on GPU')
  parser.add_argument('filepath')
  parser.add_argument('pattern')
  parser.add_argument('-n', dest='n', action='store_false', help='print only number of occurrence')
  args = parser.parse_args(sys.argv[1:])
  np.set_printoptions(threshold=np.inf)
  result = 0
  for pattern in build_patterns(args.pattern):
    result += find_pattern(pattern, args.filepath, args.n)
  print(result)


if __name__ == '__main__':
  main()
