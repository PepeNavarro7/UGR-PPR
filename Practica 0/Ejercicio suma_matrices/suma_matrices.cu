#include <iostream>
#include <fstream>
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;
const int THREADS_PER_BLOCK=16;

__global__ void MatAdd( float *A, float *B, float *C, int N){
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // Compute row index -> horizontal
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // Compute column index -> vertical
  int index=y*N+x; // Compute global 1D index 
  // y*N+x porque los threads estan traspuestos
  if (x < N && y < N)
	  C[index] = A[index] + B[index]; // Compute C element
}

int main(int argc, char* argv[]){ 
  if (argc == 2){
    const int TAM_MATRIZ = atoi(argv[1]);
    if(TAM_MATRIZ < 46340){
      const int NN=TAM_MATRIZ*TAM_MATRIZ;
      /* pointers to host memory */
      /* Allocate arrays A, B and C on host*/
      float * A = (float*) malloc(NN*sizeof(float));
      float * B = (float*) malloc(NN*sizeof(float));
      float * C = (float*) malloc(NN*sizeof(float));

      /* pointers to device memory */
      float *A_d, *B_d, *C_d;
      /* Allocate arrays a_d, b_d and c_d on device*/
      cudaMalloc ((void **) &A_d, sizeof(float)*NN);
      cudaMalloc ((void **) &B_d, sizeof(float)*NN);
      cudaMalloc ((void **) &C_d, sizeof(float)*NN);

      /* Initialize arrays a and b */
      for (int i=0; i<TAM_MATRIZ;i++){
        for (int j=0;j<TAM_MATRIZ;j++){
          A[i*TAM_MATRIZ+j]=(float) i; 
          B[i*TAM_MATRIZ+j]=(float) (1-i);
        }
      }

      /* Copy data from host memory to device memory */
      cudaMemcpy(A_d, A, sizeof(float)*NN, cudaMemcpyHostToDevice);
      cudaMemcpy(B_d, B, sizeof(float)*NN, cudaMemcpyHostToDevice);

      /* Compute the execution configuration */
      dim3 threadsPerBlock (THREADS_PER_BLOCK,THREADS_PER_BLOCK); // bloques de 16*16=256 threads
      dim3 numBlocks( ceil ((float)(TAM_MATRIZ)/threadsPerBlock.x), ceil ((float)(TAM_MATRIZ)/threadsPerBlock.y) );
      // grids de N/16 * N/16 = N²/256 bloques      
      
      auto start = high_resolution_clock::now();
      // Kernel Launch
      MatAdd <<<numBlocks, threadsPerBlock>>> (A_d, B_d, C_d, TAM_MATRIZ);
      cudaDeviceSynchronize();
      auto stop = high_resolution_clock::now();

      auto duration = duration_cast<microseconds>(stop - start);
      double micro = duration.count();
      double milli = micro / 1000.0;
      /* Copy data from deveice memory to host memory */
      cudaMemcpy(C, C_d, sizeof(float)*NN, cudaMemcpyDeviceToHost);

      /* Print c */
      /*for (int i=0; i<TAM_MATRIZ;i++)
        for (int j=0;j<TAM_MATRIZ;j++)
          cout <<"C["<<i<<","<<j<<"]="<<C[i*TAM_MATRIZ+j]<<endl;
      */
      cout << "Suma eficiente de 2 matrices de " << TAM_MATRIZ << "*" << TAM_MATRIZ << " elementos." <<endl;
      cout << "Kernel Execution Time = " << milli << " milliseconds." << endl;
      
      /* Free the memory */
      free(A); free(B); free(C);
      cudaFree(A_d); cudaFree(B_d);cudaFree(C_d);
    } else {
      cout << "La matriz que pides es demasiado grande." << endl;
    }
  } else {
    cout << "Introducir tamaño de la matriz" << endl;
  }
}
