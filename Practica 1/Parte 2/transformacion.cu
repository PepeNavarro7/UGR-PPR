#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;

__global__ void CSinCompartida(float *A, float *B, float *C){
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  int istart = blockIdx.x * blockDim.x;
  int iend  = istart + blockDim.x;
  C[index]=0.0;
  for (int j=istart; j<iend; j++){
    C[index]+= fabs((index* B[j]-A[j]*A[j])/((index+2)*max(A[j],B[j]))); 
  }   
}

__global__ void CConCompartida(float *A, float *B, float *C){
  extern __shared__ float sdata[]; // el bloque tiene de tamaño 2xblockDim
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int i = threadIdx.x;
  int Bsize = blockDim.x;
  // bloque de 0 a Bsize-1 tiene el bloque de A
  // de Bsize a 2*Bsize-1 tiene el bloque de B
  sdata[i] = A[index];
  sdata[i+Bsize] = B[index];
  __syncthreads();

  /* Rellenamos el bloque para que en el cálculo de la sumatoria, en vez de hacer accesos a 
  A[] y B[], sean a sdata[] -> A[index] = sdata[j], B[index]=sdata[j+Bsize] */
  C[index]=0.0;
  for (int j=0; j<Bsize; j++){
    C[index]+= fabs( 
      (index * sdata[j+Bsize] - sdata[j] * sdata[j])
    / ( (index+2) * max(sdata[j],sdata[j+Bsize]) ) ); 
  }  
}
__global__ void reduceMax(float * V_in, float * V_out) {
	extern __shared__ float sdata[]; // sdata de tamanio Bsize

	int tid = threadIdx.x; // indice de la hebra dentro del bloque
	int index = blockIdx.x * blockDim.x + threadIdx.x; // indice total de la hebra
	sdata[tid] = V_in[index]; // cada hebra introduce su propio indice en el array
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1){ // s = s/2
	  if (tid < s)
      sdata[tid]=max(sdata[tid],sdata[tid+s]); 	
	  __syncthreads();
	}
	if (tid == 0) // Solo la primera hebra (la que tiene el minimo definitivo) copia el valor
    V_out[blockIdx.x] = sdata[0];
}

//**************************************************************************
int main(int argc, char *argv[]){
  //Llamada
  if (argc != 3){ 
    cout << "Uso: transformacion Num_bloques Tam_bloque  "<<endl;
    return(0);
  }
  int NBlocks = atoi(argv[1]), Bsize= atoi(argv[2]);
  if(Bsize != 64 && Bsize != 128 && Bsize != 256){
    cout << "Tam_bloque debe ser 64, 128 o 256" <<endl;
    return(0);
  }
  const int N=NBlocks*Bsize;

  // PARTE DE LA CPU
  // pointers to host memory
  float *A_host, *B_host, *C_host,*D_host;

  // Allocate arrays a, b and c on host
  A_host = new float[N];
  B_host = new float[N];
  C_host = new float[N];
  D_host = new float[NBlocks];

  float mx; // maximum of C

  // Initialize arrays A and B
  for (int i=0; i<N;i++){ 
    A_host[i]= (float) (1.5*(1+(5*i)%7)/(1+i%5));
    B_host[i]= (float) (2.0*(2+i%5)/(1+i%7));    
  }

  // Time measurement  
  double t1_CPU=clock();

  // Compute C[i]
  for (int k=0; k<NBlocks;k++){ 
    int istart=k*Bsize;
    int iend  =istart+Bsize;
    for (int i=istart; i<iend;i++){ 
      C_host[i]=0.0;
      for (int j=istart; j<iend;j++){
        C_host[i]+= fabs((i* B_host[j]-A_host[j]*A_host[j])/((i+2)*max(A_host[j],B_host[j]))); 
      }   
    }
  }
  double t2_CPU=clock();
  t2_CPU=(t2_CPU-t1_CPU)/CLOCKS_PER_SEC;

  // Compute mx=max{Ci}
  mx=C_host[0];
  for (int i=0; i<N;i++){ 
    //cout << C_host[i] << " ";
    mx=max(C_host[i],mx);
  }
  
  // Compute D 
  for (int k=0; k<NBlocks;k++){ 
    int istart=k*Bsize;
    int iend  =istart+Bsize;
    D_host[k]=0.0;
    for (int i=istart; i<iend;i++){ 
      D_host[k]+=C_host[i];
    }
    D_host[k]/=Bsize;
  }

  //********************************************************************
  // PARTE GPU SIN MEMORIA COMPARTIDA
  // Preparacion de los punteros
  float *A_dev_sin, *B_dev_sin, *C_dev_sin;
  //float *C_host_sin = new float[N];

  cudaMalloc ((void **) &A_dev_sin, sizeof(float)*N);
  cudaMalloc ((void **) &B_dev_sin, sizeof(float)*N);
  cudaMalloc ((void **) &C_dev_sin, sizeof(float)*N);

  cudaMemcpy(A_dev_sin, A_host, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_dev_sin, B_host, sizeof(float)*N, cudaMemcpyHostToDevice);

  // Calculo de C
  double t1_SIN=clock();
  CSinCompartida<<< NBlocks, Bsize >>>(A_dev_sin, B_dev_sin, C_dev_sin);
  cudaDeviceSynchronize();
  double t2_SIN=clock();
  t2_SIN=(t2_SIN-t1_SIN)/CLOCKS_PER_SEC;

  /*cudaMemcpy(C_host_sin, C_dev_sin, N*sizeof(float),cudaMemcpyDeviceToHost);
  for(int i=0; i<N; ++i){
    cout << C_host_sin[i] << " ";
  }*/

  //********************************************************************
  // GPU CON MEMORIA COMPARTIDA
  // Preparacion de los punteros
  float *A_dev_con, *B_dev_con, *C_dev_con;
  //float *C_host_con = new float[N];

  cudaMalloc ((void **) &A_dev_con, sizeof(float)*N);
  cudaMalloc ((void **) &B_dev_con, sizeof(float)*N);
  cudaMalloc ((void **) &C_dev_con, sizeof(float)*N);
  

  cudaMemcpy(A_dev_con, A_host, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_dev_con, B_host, sizeof(float)*N, cudaMemcpyHostToDevice);

  // Calculo de C
  double t1_CON=clock();
  CConCompartida<<< NBlocks, Bsize, 2*Bsize >>>(A_dev_con, B_dev_con, C_dev_con);
  cudaDeviceSynchronize();
  double t2_CON=clock();
  t2_CON=(t2_CON-t1_CON)/CLOCKS_PER_SEC;

  /*cudaMemcpy(C_host_con, C_dev_con, N*sizeof(float),cudaMemcpyDeviceToHost);
  for(int i=0; i<N; ++i){
    cout << C_host_con[i] << " ";
  }*/


  //*************************************************************
  // Otros calculos
  float *max_device, *D_dev_con;
  float *max_host = new float[NBlocks];
  cudaMalloc ((void **) &max_device, sizeof(float)*NBlocks);
  cudaMalloc ((void **) &D_dev_con, sizeof(float)*NBlocks);

  reduceMax<<< NBlocks, Bsize, Bsize >>>(C_dev_con, max_device);
  cudaDeviceSynchronize();
  cudaMemcpy(max_host, max_device, NBlocks*sizeof(float),cudaMemcpyDeviceToHost);

  float max_GPU = max_host[0];
  for(int i=1; i<NBlocks; ++i){
    max_GPU = max(max_GPU,max_host[i]);
    //cout << max_host[i] << " ";
  }


  //*************************************************************
  // Pantalla
  cout << "N = " << N << " = " << NBlocks << "x" << Bsize <<endl;
  cout << "Max_CPU -> " << mx << endl;
  cout << "T_CPU -> " << t2_CPU << endl;
  cout << "T_SIN -> " << t2_SIN << endl;
  cout << "T_CON -> " << t2_CON << endl;
  cout << "Max_GPU -> " << max_GPU << endl;


  //* Free the memory */
  delete(A_host); delete(B_host); delete(C_host); delete(D_host);
  delete(max_host);
  //delete(C_host_sin);
  //delete(C_host_con);
  cudaFree(A_dev_sin);
  cudaFree(B_dev_sin);
  cudaFree(C_dev_sin);
  cudaFree(A_dev_con);
  cudaFree(B_dev_con);
  cudaFree(C_dev_con);
  cudaFree(max_device);
  cudaFree(D_dev_con);
}