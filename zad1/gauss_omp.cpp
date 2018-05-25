#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <omp.h>

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
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
     char* inputImageName = argv[2]; // Nazwa pliku wejsciowego
     char* outputImageName = argv[3]; // nazwa pliku wyjsciwego
     int n_thr = atoi(argv[1]);
     Mat image, bluredImage;
     image = imread( inputImageName, 1 ); // wczytywanie obrazu
     if( argc < 2 || !image.data )
     {
       printf( " No image data \n " );
       return -1;
     }
     bluredImage = image.clone();
     Vec3i value;
     //Rozpoczecie rozmywania
     clock_gettime(CLOCK_MONOTONIC, &start);
     #pragma omp parallel num_threads(n_thr) default(none) private(value) shared(image, bluredImage, gaussTab, gaussSum)
     {
         #pragma omp for schedule(static) 
         for (int row = 2; row < bluredImage.rows-2; row++)
         {
           for (int col = 2; col < bluredImage.cols-2; col++)
           {
                
                //Obliczanie maski

                for (int mask_row = 0; mask_row < 5; mask_row++)
                {
                    for (int mask_col = 0; mask_col < 5; mask_col++)
                    {
                        value += (Vec3i)image.at<Vec3b>(row+mask_row-2,col+mask_col-2) * gaussTab[mask_row*4+mask_col];
                    }
                }            
                bluredImage.at<Vec3b>(row, col) = (Vec3b)(value/gaussSum);
                value = 0, 0, 0;
           }
        }
    }
     clock_gettime(CLOCK_MONOTONIC, &finish);
     elapsed = (finish.tv_sec - start.tv_sec);
     elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
     //Koniec rozmywania
     cout<< "Czas: " << elapsed * 1000 << "ms" << std::endl;
     imwrite( outputImageName, bluredImage );



     // //Wyswietl obraz przed i po rozmyciu
     // namedWindow( inputImageName, WINDOW_AUTOSIZE );
     // namedWindow( outputImageName, WINDOW_AUTOSIZE );
     // imshow( inputImageName, image );
     // imshow( outputImageName, bluredImage );
     // waitKey(0);
     return 0;
}
