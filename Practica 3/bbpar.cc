/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

using namespace std;

unsigned int NCIUDADES;


int main (int argc, char **argv) {
	int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc != 3 && rank == 0){
        cerr << "La sintaxis es: bbseq <tama�o> <archivo>" << endl;
		exit(1);
    }
	NCIUDADES = atoi(argv[1]);

	int** matriz; // El puntero es común, pero la reserva la hace id0
	if(rank==0){
		matriz = reservarMatrizCuadrada(NCIUDADES);
	}

	tNodo	nodo,         // nodo a explorar
			lnodo,        // hijo izquierdo
			rnodo,        // hijo derecho
			solucion;     // mejor solucion
	bool activo,        // condicion de fin
		nueva_U;       // hay nuevo valor de cota superior
	int  U;             // valor de cota superior
	int iteraciones = 0;
	tPila pila;         // pila de nodos a explorar

	U = INFINITO;                  // inicializa cota superior
	InicNodo (&nodo);              // inicializa estructura nodo
	if(rank==0){
		LeerMatriz (argv[2], matriz);    // lee matriz de fichero
		activo = !Inconsistente(matriz); // Se examina si la matriz es correcta (tiene solucion o no)
	}
	
	if(rank=0){
		cout << "Holi";
	double t=MPI::Wtime();
	while (activo) {       // ciclo del Branch&Bound
		Ramifica (&nodo, &lnodo, &rnodo, matriz);		
		nueva_U = false;
		if (Solucion(&rnodo)) {
			if (rnodo.ci() < U) {    // se ha encontrado una solucion mejor
				U = rnodo.ci();
				nueva_U = true;
				CopiaNodo (&rnodo, &solucion);
			}
		}
		else {                    //  no es un nodo solucion
			if (rnodo.ci() < U) {     //  cota inferior menor que cota superior
				if (!pila.push(rnodo)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(matriz);
					exit (1);
				}
			}
		}
		if (Solucion(&lnodo)) {
			if (lnodo.ci() < U) {    // se ha encontrado una solucion mejor
				U = lnodo.ci();
				nueva_U = true;
				CopiaNodo (&lnodo,&solucion);
			}
		}
		else {                     // no es nodo solucion
			if (lnodo.ci() < U) {      // cota inferior menor que cota superior
				if (!pila.push(lnodo)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(matriz);
					exit (1);
				}
			}
		}
		if (nueva_U) pila.acotar(U);
		activo = pila.pop(nodo);
		iteraciones++;
	}
        t=MPI::Wtime()-t;
        //MPI::Finalize();
	printf ("Solucion: \n");
	EscribeNodo(&solucion);
        cout<< "Tiempo gastado= "<<t<<endl;
	cout << "Numero de iteraciones = " << iteraciones << endl << endl;
	liberarMatriz(matriz);
	}
	MPI_Finalize();
}


