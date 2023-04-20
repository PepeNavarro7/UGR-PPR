#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

using namespace std;

//**************************************************************************
__global__ void floyd_kernel1D(int * M, const int nverts, const int k) {
    int ij = threadIdx.x + blockDim.x * blockIdx.x; // indice global de la hebra
    int i= ij / nverts; // fila
    int j= ij - i * nverts; // columna
    if (i<nverts && j< nverts) {
		int Mij = M[ij];
		if (i != j && i != k && j != k) {
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
			Mij = (Mij > Mikj) ? Mikj : Mij;
			M[ij] = Mij;
		}
  	}
}
//**************************************************************************
//**************************************************************************
__global__ void floyd_kernel2D(int * M, const int nverts, const int k) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;  // Compute row index -> horizontal
	int y = blockIdx.y * blockDim.y + threadIdx.y;  // Compute column index -> vertical
	int index=x*nverts+y; // Compute global 1D index 
    if (x<nverts && y< nverts) {
		int Mxy = M[index];
		if (x != y && x != k && y != k) {
			int Mxky = M[x * nverts + k] + M[k * nverts + y];
			Mxy = (Mxy > Mxky) ? Mxky : Mxy;
			M[index] = Mxy;
		}
  	}
}
//**************************************************************************
//**************************************************************************
__global__ void reduceSum(int * V_in, long int * V_out, const int nverts) {
	extern __shared__ float sdata[]; // vector que comparte el bloque

	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x; // indice global de la hebra
	int i= index / nverts; // fila
    int j= index - i * nverts; // columna
	sdata[tid] = ((i < nverts)&&(j<nverts) ? V_in[index] : 0); // copiamos en sdata nuestra info
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1){ // s = s/2
		if (tid < s)
	        sdata[tid]=sdata[tid]+sdata[tid+s]; 	
		__syncthreads();
	}
	if (tid == 0) 
           V_out[blockIdx.x] = sdata[0];
}
//**************************************************************************
//**************************************************************************
// ************  MAIN FUNCTION *********************************************
int main (int argc, char *argv[]) {
    double time, Tcpu, Tgpu1;
	

    if (argc != 3) {
	    cerr << "Sintaxis: " << argv[0] << "<blocksize> <archivo de grafo>" << endl;
		return(-1);
	}
	const int blocksize = atoi(argv[1]);

    //Get GPU information
    int num_devices,devID;
    cudaDeviceProp props;
    cudaError_t err;

	err=cudaGetDeviceCount(&num_devices);
	if (err != cudaSuccess){
		cerr << "ERROR detecting CUDA devices......" << endl; exit(-1);
	}	
	    
	for (int i = 0; i < num_devices; i++) {
	    devID=i;
	    err = cudaGetDeviceProperties(&props, devID);
        if (err != cudaSuccess) {
		  cerr << "ERROR getting CUDA devices" << endl;
	    }
	}
	devID = 0;

	err = cudaSetDevice(devID); 
    if(err != cudaSuccess) {
		cerr << "ERROR setting CUDA device" <<devID<< endl;
	}

	// Declaration of the Graph object
	Graph G;
	
	// Read the Graph
	G.lee(argv[2]);

	const int nverts = G.vertices;
	const int niters = nverts;
	const int nverts2 = nverts * nverts;

	int *c_Out_M = new int[nverts2];// puntero en host a un array de tamaño N^2
	const int size_int = nverts2*sizeof(int);
	int * d_In_M = NULL; // reserva de puntero en memoria device

	err = cudaMalloc((void **) &d_In_M, size_int);
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}
    // Get the integer 2D array for the dense graph
	int *A = G.Get_Matrix();

    //**************************************************************************
	// GPU phase
	//**************************************************************************
	
    time=clock();

	// Copiamos la matriz del host al device, en la memoria que reservamos antes
	err = cudaMemcpy(d_In_M, A, size_int, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 

    // Main Loop
	// se lanza el kernel N veces
	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	 	int threadsPerBlock = blocksize;
	 	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;
        // Kernel Launch
	    floyd_kernel1D <<<blocksPerGrid,threadsPerBlock >>>(d_In_M, nverts, k);
	    err = cudaGetLastError();

	    if (err != cudaSuccess) {
	  	    fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	    exit(EXIT_FAILURE);
		}
	}
	// Se copia de vuelta el array cuadrado
	err =cudaMemcpy(c_Out_M, d_In_M, size_int, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 

	Tgpu1=(clock()-time)/CLOCKS_PER_SEC;
	
	//***********************************************************************
	// Parte bidimensional
	//***********************************************************************

	double Tgpu2;
	int *array_host = new int[nverts2]; // puntero en host a un array de tamaño N^2
	int *array_device = NULL; // puntero en memoria device

	// Reservamos la memoria en device
	err = cudaMalloc((void **) &array_device, size_int);
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}

	time=clock();

	// Copiamos la matriz del host al device, en la memoria que reservamos antes
	err = cudaMemcpy(array_device, A, size_int, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 
	// bloques de 8*8, 16*16 o 32*32 hebras, para 64, 256 o 1024 hebras por bloque
	dim3 threadsPerBlock (sqrt(blocksize),sqrt(blocksize));
	// grid bidimensional
	dim3 blocksPerGrid( ceil ((float)(nverts)/threadsPerBlock.x), ceil ((float)(nverts)/threadsPerBlock.y) );
	// se lanza el kernel N veces
	for(int k = 0; k < niters; k++) {
        // Kernel Launch
	    floyd_kernel2D <<<blocksPerGrid,threadsPerBlock >>>(array_device, nverts, k);
	    err = cudaGetLastError();
	    if (err != cudaSuccess) {
	  	    fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	    exit(EXIT_FAILURE);
		}
	}
	// Se copia de vuelta el array cuadrado
	err =cudaMemcpy(array_host, array_device, size_int, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 

	Tgpu2=(clock()-time)/CLOCKS_PER_SEC;


    //**************************************************************************
	// CPU phase
	//**************************************************************************

	time=clock();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < niters; k++) {
          kn = k * nverts;
	  for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++)
	       			if (i!=j && i!=k && j!=k){
			 	    inj = in + j;
			 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
	       }
	   }
	}
  
  Tcpu=(clock()-time)/CLOCKS_PER_SEC;
	cout << argv[2] << endl;
	cout << "TCPU -> " << Tcpu << endl;
	cout << "TGPU1 -> " << Tgpu1 << endl;
	cout << "SGPU1 -> " << Tcpu / Tgpu1 << endl;
	cout << "TGPU2 -> " << Tgpu2 << endl;
	cout << "SGPU2 -> " << Tcpu / Tgpu2 << endl;

  
  bool errors=false;
  // Error Checking (CPU vs. GPU)
  for(int i = 0; i < nverts; i++)
    for(int j = 0; j < nverts; j++)
       	if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0){
			//cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;
		  	errors=true;
		}


	if (errors)
		cout<< "Errors found in GPU1"<<endl;

	bool errors2=false;
	// Error Checking (CPU vs. GPU2)
	for(int i = 0; i < nverts; i++)
		for(int j = 0; j < nverts; j++)
			if (abs(array_host[i*nverts+j] - G.arista(i,j)) > 0){
				//cout << "Error (" << i << "," << j << ")   " << array_host[i*nverts+j] << "..." << G.arista(i,j) << endl;
				errors2=true;
			}

	if (errors2)
		cout<< "Errors found in GPU2"<<endl;

	

	//**************************************************
	// Block reduction on GPU to obtain partial minimums
	//**************************************************
	int hebrasReduccion = blocksize;
	int bloquesReduccion = (nverts2 + hebrasReduccion - 1) / hebrasReduccion;

	// Reduction vector on CPU
	long int *reduccionHost = new long int [bloquesReduccion];

	// Reduction vector on GPU
	long int *reduccionDevice; 
	cudaMalloc ((void **) &reduccionDevice, sizeof(long int)*bloquesReduccion);

	int memoriaCompartida = hebrasReduccion*sizeof(int);
	// Kernel launch to compute Reduction Vector
	reduceSum<<<bloquesReduccion, hebrasReduccion, memoriaCompartida>>>(d_In_M,reduccionDevice, nverts);

	/* Copy data from device memory to host memory */
	cudaMemcpy(reduccionHost, reduccionDevice, bloquesReduccion*sizeof(int),cudaMemcpyDeviceToHost);

	// Perform final reduction in CPU
	long int aux = 0;
	//cout<<"Performing final reduction of the partial results on CPU:"<<endl;
	for (int i=0; i<bloquesReduccion; i++) {
		aux+=reduccionHost[i]; 
	}
	aux = aux/nverts2;
	cout<<"Media -> "<<aux<<endl;

	cudaFree(d_In_M);
	cudaFree(array_device);
	cudaFree(reduccionDevice);
}
