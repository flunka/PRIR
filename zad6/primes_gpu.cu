#include <iostream>
#include<curand.h>
#include<curand_kernel.h>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <fstream>


using namespace std;
#define PRECISION 10000
#define BLOCKS_NUBMER 4096
#define THREADS_NUMBER 1

__device__ float generate( curandState* globalState, int ind ) 
{
  //int ind = threadIdx.x;
  curandState localState = globalState[ind];
  float RANDOM = curand_uniform( &localState );
  globalState[ind] = localState;
  return RANDOM;
}
// Szybkie Potegowanie modulo
//
// www.algorytm.org
// (c)2006 Tomasz Lubinski
//
__device__ int power_modulo_fast(long a, long b, long m)
{
    long i;
    long result = 1;
    long  x = a%m;
     
    for (i=1; i<=b; i<<=1)
    {
        x %= m;
        if ((b&i) != 0)
        {
            result *= x;
            result %= m;
        }
        x *= x;
    }
     
    return result;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
  int id = blockIdx.x;
  int sequence = id;
  int offset = 0;
  curand_init ( seed, sequence, offset, &state[id] );
}

__global__ void kernel(int p, bool* prime, curandState* globalState)
{ 
  int i = blockIdx.x;
  int a = 0;
  while(*prime && i < PRECISION)
  {
    a = (generate(globalState, i%BLOCKS_NUBMER) * (p-2))+1;
    if(power_modulo_fast(a, p-1, p) == 1)
    {
        i += BLOCKS_NUBMER;
    }
    else
    {
        *prime = false;
    }
  }

}

int main( int argc, char** argv ) 
{

  if( argc != 2 )
  {
      cout<< "One argument is required!" <<endl;
      cout<<"To run program type: ./primes_gpu file" << endl 
      << "file - path to file with numbers" << endl;
      return 1;
  }

  struct timespec start, finish;
  double elapsed;
  long tmp;
  vector<unsigned long> numbers;
  vector<bool> output;
  
  ifstream file;
  file.open(argv[1]);

  if (!file)
  {
      cout << "\nError opening file.\n";
      return 1;
  }
  //Reading numbers form file
  while(file >> tmp)
  {
      numbers.push_back(tmp);
  }
  file.close();
  output.resize(numbers.size());
  
  int p = 15487317;
  curandState* devStates;
  cudaMalloc ( &devStates, BLOCKS_NUBMER*sizeof( curandState ) );

  // setup seeds
  setup_kernel <<< BLOCKS_NUBMER, THREADS_NUMBER >>> ( devStates,unsigned(time(NULL)) );

  

  bool* prime;
  bool init = true;
  bool result = true;
  cudaMalloc((void**) &prime, sizeof(bool));
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < numbers.size(); i++)
  {
    p = numbers.at(i);
    cudaMemcpy(prime, &init, sizeof(bool), cudaMemcpyHostToDevice);
    kernel<<<BLOCKS_NUBMER,THREADS_NUMBER>>> (p, prime, devStates);
    cudaMemcpy(&result, prime, sizeof(bool), cudaMemcpyDeviceToHost);
    output.at(i)=result;
  }
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
  cout<< "Time: " << elapsed * 1000 << "ms" << std::endl;

  //Print the result
  for (int i = 0; i < numbers.size(); i++)
  {
      if (output.at(i))
      {
          cout << numbers.at(i) << ": prime" << endl;
      }
      else
      {
          cout << numbers.at(i) << ": composite" << endl;
      }
  }

  cudaFree(devStates);
  cudaFree(prime);
  return 0;
}