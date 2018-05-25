#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <omp.h>

using namespace std;
using namespace cv;

__global__ void kernel(int rows, int columns, int *red, int *green, int *blue, int *red_result, int *green_result, int *blue_result, int *gaussTab, int gaussSum)
{ 

  int tid = blockIdx.x*blockDim.x + threadIdx.x + (2*columns);
   
  if(tid<columns*rows - (2*columns))
  {
    //Obliczanie maski
    if ((tid % columns) > 2 && (tid % columns) < columns -2)
    {
    	red_result[tid] = 0;
      green_result[tid] = 0;
      blue_result[tid] = 0;
      for (int mask_row = 0; mask_row < 5; mask_row++)
      {
        for (int mask_col = 0; mask_col < 5; mask_col++)
        {
          red_result[tid] += red[tid+(mask_row-2)*columns+mask_col-2] * gaussTab[mask_row*4+mask_col];
          green_result[tid] += green[tid+(mask_row-2)*columns+mask_col-2] * gaussTab[mask_row*4+mask_col];
          blue_result[tid] += blue[tid+(mask_row-2)*columns+mask_col-2] * gaussTab[mask_row*4+mask_col];
        }
      }           
      red_result[tid] /= gaussSum;
      green_result[tid] /= gaussSum;
      blue_result[tid] /= gaussSum;

      if(red_result[tid]>255)red_result[tid] = 255;
      if(green_result[tid]>255)green_result[tid] = 255;
      if(blue_result[tid]>255)blue_result[tid] = 255;
      
    }
        
  }
}

int main( int argc, char** argv )
{
	int threads_number = 4;
  struct timespec start, finish;
  double elapsed;  
  int gaussTab[25] = {1,1,2,1,1,
                      1,2,4,2,1,
                      2,4,8,4,2,
                      1,2,4,2,1,
                      1,1,2,1,1};
  int gaussSum = 0;
  for (int i = 0; i < 25; i++)
  {
     gaussSum += gaussTab[i];
  }
  char* inputImageName = argv[1]; // Nazwa pliku wejsciowego
  char* outputImageName = argv[2]; // nazwa pliku wyjsciwego
  
  Mat image;

  image = imread( inputImageName, 1 ); // wczytywanie obrazu
  if( argc < 2 || !image.data )
  {
    cout << " No image data \n " << endl;
    return -1;
  }
  int *red;
  int *green;
  int *blue;

  int *red_result;
  int *green_result;
  int *blue_result;

  int *dev_red;
  int *dev_green;
  int *dev_blue;

  int *dev_red_result;
  int *dev_green_result;
  int *dev_blue_result;

  int *dev_gaussTab;
  Mat3b bluredImage(image.rows,image.cols);
  long size = image.rows*image.cols;

  red = (int*)malloc(size*sizeof(int));
  green = (int*)malloc(size*sizeof(int));
  blue = (int*)malloc(size*sizeof(int));

  red_result = (int*)malloc(size*sizeof(int));
  green_result = (int*)malloc(size*sizeof(int));
  blue_result = (int*)malloc(size*sizeof(int));

  cudaMalloc( &dev_red, size*sizeof(int));
  cudaMalloc( &dev_green, size*sizeof(int));
  cudaMalloc( &dev_blue, size*sizeof(int));
  cudaMalloc((void**) &dev_gaussTab, 25*sizeof(int));

  cudaMalloc( &dev_red_result, size*sizeof(int));
  cudaMalloc( &dev_green_result, size*sizeof(int));
  cudaMalloc( &dev_blue_result, size*sizeof(int));

  for (int i = 0; i < image.rows; i++)
  {
    for (int j = 0; j < image.cols; j++)
    {
      blue[i*image.cols+j] = image.at<Vec3b>(i,j).val[0];
      green[i*image.cols+j] = image.at<Vec3b>(i,j).val[1];
      red[i*image.cols+j] = image.at<Vec3b>(i,j).val[2];
    }
  }
  cudaMemcpy( dev_red, red, size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( dev_green, green, size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( dev_blue, blue, size*sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy( dev_red_result, red, size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( dev_green_result, green, size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( dev_blue_result, blue, size*sizeof(int), cudaMemcpyHostToDevice);


  cudaMemcpy( dev_gaussTab, gaussTab, 25*sizeof(int), cudaMemcpyHostToDevice);
  //Rozpoczecie rozmywania
  
  clock_gettime(CLOCK_MONOTONIC, &start);
   
  kernel<<<(size+threads_number-1)/threads_number,threads_number>>>(image.rows, image.cols,  dev_red,dev_green,dev_blue,dev_red_result,dev_green_result,dev_blue_result,dev_gaussTab,gaussSum);
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
  //Koniec rozmywania
  cout<< "Czas: " << elapsed * 1000 << "ms" << std::endl;
  cudaMemcpy( red_result, dev_red_result, size*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy( green_result, dev_green_result, size*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy( blue_result, dev_blue_result, size*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < image.rows; i++)
  {
    for (int j = 0; j < image.cols; j++)
    {      
      bluredImage.at<Vec3b>(i,j).val[0] = blue_result[i*image.cols+j];
      bluredImage.at<Vec3b>(i,j).val[1] = green_result[i*image.cols+j];
      bluredImage.at<Vec3b>(i,j).val[2] = red_result[i*image.cols+j];
    }
  }

  imwrite( outputImageName, bluredImage );


  cudaFree(dev_red);
  cudaFree(dev_green);
  cudaFree(dev_blue);
  cudaFree(dev_red_result);
  cudaFree(dev_green_result);
  cudaFree(dev_blue_result);
  cudaFree(dev_gaussTab);
  free(red);
  free(green);
  free(blue);
  free(red_result);
  free(green_result);
  free(blue_result);
  //Wyswietl obraz przed i po rozmyciu
  // namedWindow( inputImageName, WINDOW_AUTOSIZE );
  // namedWindow( outputImageName, WINDOW_AUTOSIZE);
  // imshow( inputImageName, image );
  // imshow( outputImageName, bluredImage );
  // waitKey(0);
  return 0;
}
