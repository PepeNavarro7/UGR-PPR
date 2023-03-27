#include <iostream>
#include <fstream>
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;
const int THREADS_PER_BLOCK=256; // Mantenemos los 256 threads por bloque (antes 16*16)

/* Ahora cada thread sumará 1 columna entera de la matriz, por lo tanto en lugar de necesitar 
TAM_MATRIZ*TAM_MATRIZ hebras (tantas hebras como elementos), ahora solo necesitaremos 
TAM_MATRIZ hebras (una hebra por columna)*/

__global__ void MatAdd( float *A, float *B, float *C, int N){
    int n_hebra = blockIdx.x * blockDim.x + threadIdx.x;  // calculamos el indice de la hebra
    // la hebra n suma la columna n completa
    C[n_hebra]=0;
    int index;
    for(int x=0;x<N;++x){ // recorremos toda la columna
        index=x*N+n_hebra; // Busco el indice del elemento de la matriz
        if (x < N && n_hebra < N){
            C[n_hebra] += A[index] + B[index]; // Compute C element
        }
    }
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
      float * C = (float*) malloc(TAM_MATRIZ*sizeof(float)); // <- C ahora almacena la suma de la fila entera

      /* pointers to device memory */
      float *A_d, *B_d, *C_d;
      /* Allocate arrays a_d, b_d and c_d on device*/
      cudaMalloc ((void **) &A_d, sizeof(float)*NN);
      cudaMalloc ((void **) &B_d, sizeof(float)*NN);
      cudaMalloc ((void **) &C_d, sizeof(float)*TAM_MATRIZ); // <- C ahora almacena la suma de la fila entera

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
      dim3 threadsPerBlock (THREADS_PER_BLOCK); // <- AHORA UNIDEMENSIONALES
      dim3 numBlocks( ceil ((float)(TAM_MATRIZ)/threadsPerBlock.x) );
      // grids de N/256 bloques
      
      auto start = high_resolution_clock::now();
      // Kernel Launch
      MatAdd <<<numBlocks, threadsPerBlock>>> (A_d, B_d, C_d, TAM_MATRIZ);
      
      cudaDeviceSynchronize();
      auto stop = high_resolution_clock::now();

      auto duration = duration_cast<microseconds>(stop - start);
      /* Copy data from deveice memory to host memory */
      cudaMemcpy(C, C_d, sizeof(float)*TAM_MATRIZ, cudaMemcpyDeviceToHost);
      double micro = duration.count();
      double milli = micro / 1000.0;

      /* Print c */
      /*for (int i=0; i<TAM_MATRIZ;i++)
        cout <<"C["<<i<<"]="<<C[i]<<endl;*/
      
      cout << "Suma POR COLUMNAS 2 matrices de " << TAM_MATRIZ << "*" << TAM_MATRIZ << " elementos." <<endl;
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
