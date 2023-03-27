#include <iostream>
#include <fstream>


using namespace std;

const int N=64; // Size of the vectors


__global__ void VecAdd( float *A, float *B, float *C, int Ntot)
{
int i=threadIdx.x;
if (i<Ntot)
  {
    C[i]=A[i]+B[i];
    printf("Thread %i::  C[%i] = A[%i]+B[%i] = %f \n",i,i,i,i,C[i] ); 
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
   for (int i=0; i<N;i++)
   {
     a[i]= (float) i;
     b[i]= -(float) i;
   }

   /* Copy data from host memory to device memory */
   cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice);
   cudaMemcpy(b_d, b, sizeof(float)*N, cudaMemcpyHostToDevice);
   cout<<"************  LAUNCHING THE KERNEL TO ADD VECTORS"<<endl<<endl;
   /* Add arrays a and b, store result in c */
   VecAdd<<< 1, N >>>(a_d, b_d, c_d, N);
   /* Copy data from deveice memory to host memory */
   cudaMemcpy(c, c_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
   cout<<endl<<"END OF EXECUTION:   SHOWING  RESULT VECTOR C:";

   
   /* Print c */
   cout<<endl<<"C = { ";
   for (int i=0; i<N;i++)
     cout<<"  "<<c[i];
   cout<<" }"<<endl;   

   /* Free the host and device  memory */
   free(a); free(b); free(c);
   cudaFree(a_d); cudaFree(b_d);cudaFree(c_d);

}
