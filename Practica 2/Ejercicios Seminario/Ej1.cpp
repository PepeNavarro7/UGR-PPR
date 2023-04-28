#include "mpi.h"
#include <iostream>
using namespace std;
 
int main(int argc, char *argv[]){
    int id, rank, size, enviar;
    MPI_Status estado;
 
    MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &id); // Obtenemos el valor de nuestro identificador

    MPI_Comm pares_impares;
    int color = id%2;
    MPI_Comm_split(MPI_COMM_WORLD, color, id, &pares_impares);
    MPI_Comm_rank(pares_impares, &rank); // rank en el nuevo comunicador
    MPI_Comm_size(pares_impares, &size); // nuevo size
 
    if(rank == 0){ // este lo ejecutan 0 y 1, que solo envian
        enviar=id;
        MPI_Send(&enviar, 1, MPI_INT, rank+1, 0, pares_impares);
    } else{ // el resto, todos reciben
        MPI_Recv(&enviar, 1, MPI_INT, rank-1, 0, pares_impares, &estado);
        cout<<"Soy el proceso "<<id<<" y he recibido "<<enviar<<endl;
        if(rank != size-1){ // y salvo los ultimos, el resto re-envian todos
            MPI_Send(&enviar, 1, MPI_INT, rank+1, 0, pares_impares);
        }
    }
    MPI_Finalize();
    return 0;
}