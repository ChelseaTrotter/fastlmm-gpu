#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cblas.h>
//#include <gsl/gsl_blas.h>

// compile as:
// export PATH=$PATH:/usr/local/cuda/bin
// nvcc cudaExample.C -I/usr/local/cuda/include -lcublas -o cudaExample

#define MAX 5120

matrixMulCPU(float *C, float *A, float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    cblas_sgemm (CblasRowMajor, 
               CblasNoTrans, CblasNoTrans, hA, wA,wB,
               1.0, B, hA, A, wB, 0.0, C, hA);
}

double read_timer() {
  struct timeval end;
  gettimeofday( &end, NULL );
  return end.tv_sec+1.e-6*end.tv_usec;
}

void fillMatrix( float *p, int n ) {
  int i;
  srand48(0);
  for( i = 0; i < n; i++ )
    p[i] = 2*drand48()-1;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            // printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}


int main( int argc, char **argv ) {
  printf("Starting\n");
  int size;
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int it;

  cublasOperation_t N = 'N';
  cublasOperation_t T = 'T';
  float one = 1., zero=0.;

  for( size = MAX; size <= MAX; size*=2 ) {

    // allocate memory on host (CPU)
    float *A = (float*) malloc( sizeof(float)*size*size );
    float *B = (float*) malloc( sizeof(float)*size*size );
    float *C = (float*) malloc( sizeof(float)*size*size );

    cudaDeviceSynchronize();
    double tInit = read_timer();

    float *dA,*dB,*dC;
    // allocate memory on device (GPU)
    cudaStat = cudaMalloc((void**)&dA, sizeof(float)*size*size);
    if(cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc((void**)&dB, sizeof(float)*size*size);
    if(cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc((void**)&dC, sizeof(float)*size*size);
    if(cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
    }

    // wait until previous CUDA commands on GPU threads have finished
    // this allows us to do the timing correctly
    cudaDeviceSynchronize();

    double tAlloc = read_timer();

    
    // initialization of CUBLAS
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
    }

    // create our test matrix on the CPU
    fillMatrix(A, size*size);
    fillMatrix(B, size*size);

    cudaDeviceSynchronize();
    double tInit2 = read_timer();


    // copy matrix to GPU, with dB the pointer to the object on the GPU
    stat = cublasSetMatrix (size, size, sizeof(float), A, size, dA, size);
    stat = cublasSetMatrix (size, size, sizeof(float), B, size, dB, size);
    stat = cublasSetMatrix (size, size, sizeof(float), C, size, dC, size);

    if(stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (dB);
      cublasDestroy(handle);
      return EXIT_FAILURE;
    }

    cudaDeviceSynchronize();
    double tTransferToGPU = read_timer();
 
    // call cublas matrix multiply (dA = dB * dB)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &one, dA, size, dB, size, &zero, dC, size );

    cudaDeviceSynchronize();
    double tMatMult = read_timer();

    // transfer matrix back to CPU
    stat = cublasGetMatrix (size, size, sizeof(float), dC, size, C, size);
    if(stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data upload failed");
      cudaFree(dC);
      cublasDestroy(handle);
      return EXIT_FAILURE;
    }
    
    cudaDeviceSynchronize();
    double tTransferFromGPU = read_timer();

    printf("====================================================\n");
    printf("Timing results for n = %d\n", size);
    printf("GPU memory allocation time: %f\n", tAlloc - tInit);
    printf("Transfer to GPU time: %f\n", tTransferToGPU - tInit2);
    printf("Matrix multiply time: %f\n", tMatMult - tTransferToGPU);
    printf("Transfer from GPU time: %f\n", tTransferFromGPU - tMatMult);
    printf("GPU total time: %f\n", tTransferFromGPU - tInit2);

    double c_start= read_timer();
    float *h_C = (float*) malloc( sizeof(float)*size*size );

    matrixMulCPU(h_C, A,B, size,size,size);
    double c_end = read_timer();
    printf("CPU calc time: %f\n", c_end - c_start);

    printDiff(h_C,C, size, size, 100, 1.0e-3);
    // free memory on GPU and CPU
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    free(A);
    free(B);
    free(C);

  }
  return EXIT_SUCCESS;
}
