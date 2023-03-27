#include "stdio.h"
#include "iostream"
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define Bsize_addition 256
#define Bsize_minimum   128

using namespace std;

//**************************************************************************
// Vector addition kernel
//**************************************************************************
__global__ void add_arrays_gpu( float *in1, float *in2, float *out, int N){
	int idx=blockIdx.x*blockDim.x+threadIdx.x; // se calcula el indice de la hebra
	if (idx<N) 
		out[idx]=in1[idx]+in2[idx];
}
//**************************************************************************



//**************************************************************************
// Vector minimum  kernel
//**************************************************************************
__global__ void reduceMin(float * V_in, float * V_out, const int N) {
	extern __shared__ float sdata[]; // revisar las palabras clave -> extern + shared

	int tid = threadIdx.x; // indice de la hebra dentro del bloque
	int i = blockIdx.x * blockDim.x + threadIdx.x; // indice total de la hebra
	sdata[tid] = ((i < N) ? V_in[i] : 100000000.0f); // cada hebra introduce su propio indice en el array
	__syncthreads();

	/* En cada iteracion divido tid (que tiene el tamaño inicial del bloque), entre 2, 
	y las hebras de la mitad izquierda comparan el valor de "su celda" con el de su par de la derecha,
	es decir, comparan tid con tid+s, y lo guardan en tid.
	En la siguiente iteracion, divido s entre 2, y divido la antigua mitad izquierda entre 2 de nuevo.
	*/ 
	for(int s = blockDim.x/2; s > 0; s >>= 1){ // s = s/2
	  if (tid < s)
	        sdata[tid]=min(sdata[tid],sdata[tid+s]); 	
	  __syncthreads();
	}
	if (tid == 0) // Solo la primera hebra (la que tiene el minimo definitivo) copia el valor
        V_out[blockIdx.x] = sdata[0];
}

//**************************************************************************
// Vector minimum  kernel
//**************************************************************************
/* En el original hay tantas hebras como elementos (N).
Cada hebra idx coge el valor idx del vector V_in de tamaño N, 
y lo guarda en el vector compartido sdata de tamaño blockDim.x en la posicion tid=ThreadIdx.x
El problema que observamos es que de las blockDim hebras (tid) que hay en el bloque, luego al iterar,
solo se utilizan la mitad de ellas. Pretendemos que se utilicen todas de primeras.

Una de dos, o pongo la mitad de hebras por bloque con los mismos bloques, 
o la misma cantidad de hebras por bloque con la mitad de bloques,
voy a optar por la primera opcion.

Este kernel se lanzará con la mitad de hebras N/2, en bloques de la mitad de tamaño,
es decir, misma cantidad de bloques de antes pero mas pequeños.
sdata sin embargo seguirá teniendo el mismo tamaño de antes
*/
__global__ void reduceMinMitadHebras(float * V_in, float * V_out, const int N) {
	extern __shared__ float sdata[]; // Vector que comparte el bloque, de tamaño blockDim * 2

	int tid = threadIdx.x; // indice de la hebra dentro del bloque
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // indice total de la hebra
	sdata[tid] = V_in[idx]; // cada hebra introduce su propio indice en el array
	// Ademas, como tenemos la mitad de hebras por bloque, introducimos otro valor
	// Acceso coalescente, cada hebra accede a idx y a idx+N/2
	sdata[tid+blockDim.x] = ( ( (idx+N/2) < N) ? V_in[(idx+N/2)] : 100000000.0f);
	__syncthreads();

	// Esto funciona igual
	for(int s = blockDim.x; s > 0; s >>= 1){ // s ahora es blockDim, es decir, sdata.size/2
	  if (tid < s)
	        sdata[tid]=min(sdata[tid],sdata[tid+s]); 	
	  __syncthreads();
	}
	if (tid == 0) // Solo la primera hebra (la que tiene el minimo definitivo) copia el valor
        V_out[blockIdx.x] = sdata[0];
}
__global__ void reduceMinMitadHebras(float * V_in, float * V_out, const int N) {
	extern __shared__ float sdata[]; // Vector que comparte el bloque, de tamaño blockDim * 2

	int tid = threadIdx.x; // indice de la hebra dentro del bloque
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // indice total de la hebra
	sdata[tid] = V_in[idx]; // cada hebra introduce su propio indice en el array
	// Ademas, como tenemos la mitad de hebras por bloque, introducimos otro valor
	// Acceso coalescente, cada hebra accede a idx y a idx+N/2
	sdata[tid+blockDim.x] = ( ( (idx+N/2) < N) ? V_in[(idx+N/2)] : 100000000.0f);
	__syncthreads();

	// Esto funciona igual
	for(int s = blockDim.x; s > 0; s >>= 1){ // s ahora es blockDim, es decir, sdata.size/2
	  if (tid < s)
	        sdata[tid]=min(sdata[tid],sdata[tid+s]); 	
	  __syncthreads();
	}
	if (tid == 0) // Solo la primera hebra (la que tiene el minimo definitivo) copia el valor
        V_out[blockIdx.x] = sdata[0];
}

//**************************************************************************
int main(int argc, char* argv[]){
	if (argc != 2){
		cout << "Funcionamiento:" << argv[0] << " N." << endl;
		exit(-1);
	}
	int N = atoi(argv[1]);

	srand(time(NULL));
	/* pointers to host memory */
	float *a, *b, *c;
	/* pointers to device memory */
	float *a_d, *b_d, *c_d;
	int i;

	/* Allocate arrays a, b and c on host*/
	a = (float*) malloc(N*sizeof(float));
	b = (float*) malloc(N*sizeof(float));
	c = (float*) malloc(N*sizeof(float));

	/* Allocate arrays a_d, b_d and c_d on device*/
	cudaMalloc ((void **) &a_d, sizeof(float)*N);
	cudaMalloc ((void **) &b_d, sizeof(float)*N);
	cudaMalloc ((void **) &c_d, sizeof(float)*N);

	/* Initialize arrays a and b */
	for (i=0; i<N;i++){
		a[i]= (float) (rand()%1000); // a[i] entre 0 y 99
		b[i]= (float) (rand()%45); // b[i] entre 0 y 44
	}


	/* Copy data from host memory to device memory */
	cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, sizeof(float)*N, cudaMemcpyHostToDevice);

	/* Compute the execution configuration */
	dim3 dimBlock(Bsize_addition);
	dim3 dimGrid ( ceil( (float(N)/(float)dimBlock.x) ) );

	/* Add arrays a and b, store result in c */
	add_arrays_gpu<<< dimGrid, dimBlock >>>(a_d, b_d, c_d, N);

	//**************************************************
	// Block reduction on GPU to obtain partial minimums
	//**************************************************
	dim3 threadsPerBlock(Bsize_minimum, 1); // por qué el 1? es unidimensional
	dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), 1);

	// Minimum vector on CPU
	float * vmin;
	vmin = (float*) malloc(numBlocks.x*sizeof(float));

	// Minimum vector  to be computed on GPU
	float *vmin_d; 
	cudaMalloc ((void **) &vmin_d, sizeof(float)*numBlocks.x);
	int smemSize = threadsPerBlock.x*sizeof(float);
	
	auto start1 = high_resolution_clock::now();
	// Kernel launch to compute Minimum Vector
	reduceMin<<<numBlocks, threadsPerBlock, smemSize>>>(c_d,vmin_d, N);
	cudaDeviceSynchronize();
	auto stop1 = high_resolution_clock::now();
	auto duration1 = duration_cast<nanoseconds>(stop1 - start1);
	double nano1 = duration1.count();
	double micro1 = nano1 / 1000.0;

	/* Copy data from device memory to host memory */
	cudaMemcpy(vmin, vmin_d, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(c, c_d, sizeof(float)*N,cudaMemcpyDeviceToHost);

	// Perform final reduction in CPU
	float min_gpu = 10000000.0f;
	//cout<<"Performing final reduction of the partial results on CPU:"<<endl;
	for (int i=0; i<numBlocks.x; i++){
		min_gpu =min(min_gpu,vmin[i]); 
		//cout<<"vmin["<<i<<"]="<<vmin[i]<<"    ";
	}
	//cout<<endl<<"... Minimum on GPU ="<<min_gpu<<endl;
	cout << "Tiempo del Kernel-1 -> "<<micro1<<" microsegundos."<< endl;

//********************************************************************************
// Reduccion con la mitad de hebras
//********************************************************************************
	dim3 threadsPerBlock2((Bsize_minimum/2), 1); // 64 hebras por bloque en lugar de 128
	dim3 numBlocks2( ceil( ( (float)N/2.0 ) / threadsPerBlock2.x ), 1); // OJO, numblocks2 = numblocks1

	// Minimum vector on CPU
	float * vmin2;
	vmin2 = (float*) malloc(numBlocks2.x*sizeof(float));

	// Minimum vector  to be computed on GPU
	float *vmin_d2; 
	cudaMalloc ((void **) &vmin_d2, sizeof(float)*numBlocks2.x);
	int smemSize2 = threadsPerBlock2.x*2*sizeof(float); // smemSize2 == smemSize1

	auto start2 = high_resolution_clock::now();
	// Kernel launch to compute Minimum Vector
	reduceMinMitadHebras<<<numBlocks2, threadsPerBlock2, smemSize2>>>(c_d,vmin_d2, N);
	cudaDeviceSynchronize();
	auto stop2 = high_resolution_clock::now();
	auto duration2 = duration_cast<nanoseconds>(stop2 - start2);
	double nano2 = duration2.count();
	double micro2 = nano2 / 1000.0;


	/* Copy data from device memory to host memory */
	cudaMemcpy(vmin2, vmin_d2, numBlocks2.x*sizeof(float),cudaMemcpyDeviceToHost);
	//cudaMemcpy(c, c_d, sizeof(float)*N,cudaMemcpyDeviceToHost); // c_d ya fue copiado

	// Perform final reduction in CPU
	float min_gpu2 = 10000000.0f;
	//cout<<"Performing final reduction2 of the partial results on CPU:"<<endl;
	for (int i=0; i<numBlocks2.x; i++){
		min_gpu2 =min(min_gpu2,vmin2[i]); 
		//cout<<"vmin2["<<i<<"]="<<vmin2[i]<<"    ";
	}
	//cout<<endl<<"... Minimum on GPU2 ="<<min_gpu2<<endl;
	cout << "Tiempo del Kernel-2 -> "<<micro2<<" microsegundos."<< endl;

	
      

      

	//***********************
	// Compute minimum on CPU
	//***********************
	float min_cpu=1000000.0f;
	for (i=0; i<N;i++)	{
		//ut<<"c["<<i<<"]="<<c[i]<<endl;
		min_cpu=min(min_cpu, c[i]);
	}
	//cout<<"Minimum on CPU="<< min_cpu<<endl;


	/* Free the memory */
	free(a); free(b); free(c); free(vmin); free(vmin2);
	cudaFree(a_d); cudaFree(b_d);cudaFree(c_d);	cudaFree(vmin_d); cudaFree(vmin_d2);
}
