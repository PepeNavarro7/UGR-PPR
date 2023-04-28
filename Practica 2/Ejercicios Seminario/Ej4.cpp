#include "mpi.h"
#include <vector>
#include <cstdlib>
#include <iostream>
using namespace std;
 
int main(int argc, char *argv[]) {
    int rank, size;
 
    MPI_Init(&argc, &argv); //iniciamos el entorno MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //obtenemos el identificador del proceso
    MPI_Comm_size(MPI_COMM_WORLD, &size); //obtenemos el numero de procesos
 
    int a;
    int b;
    if (rank == 0) {
        a = 2000;
        b = 1;
    } else {
        a = 0;
        b = 0;
    }
 
    MPI_Comm pares_impares, inverso; 
    int rank_par_impar, size_par_impar, rank_inverso;
    int color = rank % 2;
    // creamos un nuevo cominicador
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &pares_impares);
    MPI_Comm_split(MPI_COMM_WORLD, 0, size-rank, &inverso); 
 
    MPI_Comm_rank(pares_impares, &rank_par_impar);
    MPI_Comm_size(pares_impares, &size_par_impar); 
    MPI_Comm_rank(inverso, &rank_inverso); 
    
 
    //Probamos a enviar datos por distintos comunicadores
    MPI_Bcast(&a, 1, MPI_INT, 0, pares_impares); // enviamos a del 0 a los pares, y a del 1 (0 en impares) a los impares
    MPI_Bcast(&b, 1, MPI_INT, size-1, inverso); // aqui el root es el 0 (el s-1 del inverso)

    
    int local = 0;
    vector<int> repartir(size_par_impar);
    if(rank==1){
        for(int i=0; i<size_par_impar; ++i){
            repartir[i] = 28;
        }
    }
    if(rank%2 == 1)
        MPI_Scatter(&repartir[0], 1, MPI_INT, &local, 1, MPI_INT, 0, pares_impares);
    
    cout << rank << " -> local -> " << local << endl;
    
    /*
    cout << "Soy el proceso " << rank << " de " << size << " dentro de MPI_COMM_WORLD,"
    "\n\t mi rango en COMM_nuevo es " << rank_par_impar << ", de " << size_par_impar <<
    ", aqui he recibido el valor " << a <<
    ",\n\ten COMM_inverso mi rango es " << rank_inverso << " de " << size <<
    " aqui he recibido el valor " << b <<"\n"<< endl;
    */
    
    MPI_Finalize();
    return 0;
}