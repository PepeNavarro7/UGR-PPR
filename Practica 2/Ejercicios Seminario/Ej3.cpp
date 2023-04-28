#include "mpi.h"
#include <vector>
#include <cstdlib>
#include <iostream>
using namespace std;
 
int main(int argc, char *argv[]) {
    int tama, rank, size;
 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // numero de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // id del proceso
 
    // Esto lo tiene el profe para asegurarse de que son multiplos
    if (argc < 2) {
        if (rank == 0) {
            cout << "No se ha especificado numero de elementos, multiplo de la cantidad de entrada, por defecto sera " << size * 100;
            cout << "\nUso: <ejecutable> <cantidad>" << endl;
        }
        tama = size * 100;
    } else {
        tama = atoi(argv[1]);
        if (tama < size) tama = size;
        else {
            int i = 1, num = size;
            while (tama > num) {
                ++i;
                num = size*i;
            }
            if (tama != num) {
                if (rank == 0)
                    cout << "Cantidad cambiada a " << num << endl;
                tama = num;
            }
        }
    }
 
    // Hacemos los vectores
    vector<long> VectorA (tama), VectorALocal(tama/size), VectorBLocal(tama/size);

    if (rank == 0) { // Valores vector A
        for (long i = 0; i < tama; ++i) {
            VectorA[i] = i + 1; // Vector A recibe valores 1, 2, 3, ..., tama
        }
    }
 
    // Repartimos los valores de A
    // Importante el [0], si no, peta
    MPI_Scatter(&VectorA[0], tama/size, MPI_LONG, &VectorALocal[0], tama/size, MPI_LONG, 0, MPI_COMM_WORLD);

    // Damos valores al bloque local del vector B
    // El quid de la cuestion es que si hay que repartir 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 y 11
    // Se reparten 0,1,2 al 0; 3,4,5 al 1; 6,7,8 al 2; y 9,10,11 al 3
    // Entonces yo inicializo esos 3,4,5 en el vectorlocal del P1
    long istart = rank * tama/size, iend=(rank+1)*tama/size;
    for (long i = 0; i<tama/size || istart<iend; ++i, ++istart){
        VectorBLocal[i] = (istart + 1)*10;
    }
        

    // Calculamos la multiplicacion de los vectores locales
    long producto = 0;
    for (long i = 0; i < tama / size; ++i) {
        producto += VectorALocal[i] * VectorBLocal[i];
    }
    long total;
 
    // Reducimos
    MPI_Reduce(&producto, &total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
 
    if (rank == 0)
        cout << "Total = " << total << endl;

    MPI_Finalize();
    return 0;
}
