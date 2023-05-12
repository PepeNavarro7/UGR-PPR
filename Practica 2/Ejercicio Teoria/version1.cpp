#include "mpi.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;
int main(int argc, char *argv[]){

////////////////////////////////////////////////////////////////////////////////////////////
// Declaracion de variables
//////////////////////////////////////////////////////////////////////////////////////////// 
    int idProceso, numProcesadores;
    float *vectorGlobal, *vectorLocal;
    MPI_Status estado;
 
    MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &idProceso); // Obtenemos el valor de nuestro identificador
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesadores); // Y el tamaño


////////////////////////////////////////////////////////////////////////////////////////////
// Chequeo de valores
//////////////////////////////////////////////////////////////////////////////////////////// 
    if (argc <= 2) {
        if (idProceso==0)
            cout << "Debes aportar el numero de iteraciones y N en la linea de comandos."<< endl;
        MPI_Finalize();
        return 0;
    }
    const int num_iter = atoi(argv[1]); // Numero de iteraciones
    if(num_iter<1 || num_iter>10){
        if (idProceso==0)
            cout << "Deberá haber entre 1 y 10 iteraciones"<< endl;
        MPI_Finalize();
        return 0;
    }
    const int N = atoi(argv[2]); // Tamaño del lado del vector
    if (N<1){
        if (idProceso==0)
            cout << "El vector ha de tener un tamaño positivo."<< endl;
        MPI_Finalize();
        return 0;
    } 
    if (N%numProcesadores != 0){
        if (idProceso==0)
            cout << "N debe ser multiplo de np."<< endl;
        MPI_Finalize();
        return 0;
    }
    const int TAM = N/numProcesadores; // Tamaño del lado de los bloques

//////////////////////////////////////////////////////////////////////////////
// Inicializacion y scatter
//////////////////////////////////////////////////////////////////////////////

    if(idProceso == 0){
        vectorGlobal = new float[N];
        srand (static_cast <unsigned> (time(0)));

        // Rellena el vector global
        for (int i = 0; i < N; i++) // Random entre [0.0 y 1.0]
            vectorGlobal[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    vectorLocal = new float[TAM];
    // Repartimos los valores del vector global a los locales
    MPI_Scatter(vectorGlobal, TAM, MPI_FLOAT, 
        vectorLocal, TAM, MPI_FLOAT, 
        0, MPI_COMM_WORLD); 
    

//////////////////////////////////////////////////////////////////////////////
// Cálculo
//////////////////////////////////////////////////////////////////////////////

    const int proc_izq = (idProceso-1+numProcesadores)%numProcesadores, // procesador de la izquierda
        proc_der = (idProceso+1)%numProcesadores; // procesador de la derecha

    for(int i=0; i<num_iter; ++i){
        // En cada iteración, intercambiamos datos, y operamos
        float izquierda, derecha; // valores frontera

        // Enviamos el primer valor al proceso izquierda, y el ultimo al derecha
        MPI_Send(&vectorLocal[0]    , 1, MPI_FLOAT, proc_izq, 0, MPI_COMM_WORLD);
        MPI_Send(&vectorLocal[TAM-1], 1, MPI_FLOAT, proc_der, 0, MPI_COMM_WORLD);

        // Recibimos los valores frontera 
        MPI_Recv(&izquierda, 1, MPI_FLOAT, proc_izq, 0, MPI_COMM_WORLD, &estado);
        MPI_Recv(&derecha  , 1, MPI_FLOAT, proc_der, 0, MPI_COMM_WORLD, &estado);

        for(int j=0; j<TAM-1; ++j){
            int aux = vectorLocal[j];
            vectorLocal[j]=(izquierda-vectorLocal[j]+vectorLocal[j+1])/2;
            izquierda = aux;
        }
        vectorLocal[TAM-1] = (izquierda-vectorLocal[TAM-1]+derecha)/2;
    }

///////////////////////////////////////////////////////////////////////
// Gather
//////////////////////////////////////////////////////////////////////

    MPI_Gather(vectorLocal, TAM, MPI_FLOAT,
	    vectorGlobal, TAM, MPI_FLOAT, 
        0, MPI_COMM_WORLD);
    MPI_Finalize();

    if(idProceso==0){
        string st = "";
        for (int i=0; i<N; ++i){
            st+= to_string(vectorGlobal[i]) + "     ";
        }
        cout << st << endl;
        delete[] vectorGlobal;
    }    
    delete[] vectorLocal;
    return 0;
}