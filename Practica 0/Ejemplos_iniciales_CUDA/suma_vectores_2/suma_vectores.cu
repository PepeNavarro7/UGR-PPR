#include <iostream>
#include <fstream>


using namespace std;

const int N=150;

__global__ void add_arrays_gpu( float *in1, float *in2, float *out, int Ntot){
	int idx=blockIdx.x*blockDim.x+threadIdx.x; // indice de la hebra
	int i=idx*2; // Cada hebra idx se ocupa del lugar i y de i+1
	if (i<Ntot){
		out[i]=in1[i]+in2[i];
		printf("Block %i,   Thread %i::  C[%i] = A[%i]+B[%i] = %f \n",blockIdx.x,threadIdx.x,i,i,i,out[i] ); 
		i++;
		if(i<Ntot){
			out[i]=in1[i]+in2[i];
			printf("Block %i,   Thread %i::  C[%i] = A[%i]+B[%i] = %f \n",blockIdx.x,threadIdx.x,i,i,i,out[i] );
		}
	}
}

int main(int argc, char* argv[])
{
  /* pointers to host memory */
  float *a, *b, *c;
  /* pointers to device memory */
  float *a_d, *b_d, *c_d;

  /* Allocate arrays a, b and c on host*/
  a = (float*) malloc(N*sizeof(float));
  b = (float*) malloc(N*sizeof(float));
  c = (float*) malloc(N*sizeof(float));

  /* Allocate arrays a_d, b_d and c_d on device*/
  cudaMalloc ((void **) &a_d, sizeof(float)*N);
  cudaMalloc ((void **) &b_d, sizeof(float)*N);
  cudaMalloc ((void **) &c_d, sizeof(float)*N);

  /* Initialize arrays a and b */
  for (int i=0; i<N;i++){
    a[i]= (float) i;
    b[i]= -(float) i;
  }

  /* Copy data from host memory to device memory */
  cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(float)*N, cudaMemcpyHostToDevice);

  /* Compute the execution configuration */
  int block_size=64; // siempre multiplo de 32
  dim3 dimBlock(block_size);
  dim3 dimGrid ( ceil((float(N)/(float)dimBlock.x/2.0)) ); // Entre 2 porque cada hebra hace 2 operaciones

  /* Add arrays a and b, store result in c */
  add_arrays_gpu<<< dimGrid, dimBlock >>>(a_d, b_d, c_d, N);

  /* Copy data from deveice memory to host memory */
  cudaMemcpy(c, c_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

  /* Print c */
  cout<<endl<<endl<<"C = { ";
  for (int i=0; i<N;i++)
     cout<<"  "<<c[i];
  cout<<" }"<<endl;   

  // Free the memory
  free(a); free(b); free(c);
  cudaFree(a_d); cudaFree(b_d);cudaFree(c_d);

}
