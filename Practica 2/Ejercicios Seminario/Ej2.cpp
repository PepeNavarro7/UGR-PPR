#include <math.h>
#include <cstdlib> // Incluido para el uso de atoi
#include <iostream>
#include "mpi.h" 
using namespace std;
 
int main(int argc, char *argv[]) { 
    int rank, size;
    const double PI25DT = 3.141592653589793238462643;
 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    int bloques;
    if(rank == 0){
        cout<<"introduce la precision del calculo (n >= np): ";
	    cin>>bloques;
    }
    MPI_Bcast(&bloques, 1, MPI_INT, 0, MPI_COMM_WORLD); // Transmitimos el dato
    int Bsize = ceil((float)bloques/size); // bloques que se queda cada proceso (salvo el ultimo)

    const double ancho = 1.0 / (double) bloques;
    double sum_local = 0.0, sum_total;
    for (int i = rank*Bsize; i<(rank+1)*Bsize && i<bloques; ++i) {
        double x = ancho * ((double)i + 0.5);
        sum_local += (4.0 / (1.0 + x*x));
    }

    MPI_Allreduce(&sum_local, &sum_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // Para que reciban todos
    if(rank==0){
        double pi = sum_total * ancho;
        cout << "El valor aproximado de PI es: " << pi << ", con un error de " << fabs(pi - PI25DT) << endl;
    }
	MPI_Finalize();
	return 0;
 
}