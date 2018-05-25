#include <iostream>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <omp.h>
#include <vector>

using namespace std;

#define PRECISION 10000

int power_modulo_fast(long a, long b, long m);

int main( int argc, char** argv )
{
    if( argc != 3 )
    {
        cout<< "Two arguments are required!" <<endl;
        cout<<"To run program type: ./primes_omp n file" << endl << "n - number of threads" << endl 
        << "file - path to file with numbers" << endl;
        return 1;
    }
    struct timespec start, finish;
    double elapsed;
    ifstream file;
    int n_thr = atoi(argv[1]);
    vector<unsigned long> numbers;
    vector<bool> result;
    long tmp,a,j,p;
    bool prime;

    srand( time( NULL ) );
    file.open(argv[2]);
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
    //Setting size of result list equal to size of number list
    result.resize(numbers.size());
    //Start of prime test
    clock_gettime(CLOCK_MONOTONIC, &start);
    #pragma omp parallel num_threads(n_thr) default(none) shared(numbers,cout,result) private(p,j,prime,a)
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < numbers.size(); i++)
        {
            p = numbers.at(i);       
            j = 0;
            prime = true;
            /*end when p is not prime (a^(p-1) mod p != 1)
            or test passed PRECISION times*/
            while(prime && j < PRECISION)
            {
                a = (rand() % (p-1)) + 1;
                if(power_modulo_fast(a, p-1, p) == 1)
                {
                    j++;
                }
                else
                {
                    prime = false;
                }
            }
            result.at(i)=prime;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout<< "Time: " << elapsed * 1000 << "ms" << std::endl;
    //Print the result
    for (int i = 0; i < numbers.size(); i++)
    {
        if (result.at(i))
        {
            cout << numbers.at(i) << ": prime" << endl;
        }
        else
        {
            cout << numbers.at(i) << ": composite" << endl;
        }
    }

    return 0;
}

//
// Szybkie Potegowanie modulo
//
// www.algorytm.org
// (c)2006 Tomasz Lubinski
//
int power_modulo_fast(long a, long b, long m)
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