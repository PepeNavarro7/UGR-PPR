/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

unsigned int NCIUDADES;
int idProceso, numProcesos, SIGUIENTE, ANTERIOR;

MPI_Comm comunicadorCarga;

int main(int argc, char **argv){
    if (argc == 3){
        NCIUDADES = atoi(argv[1]);
    }else {
		std::cerr << "La sintaxis es: bbseq <tama�o> <archivo>" << std::endl;
		exit(1);
	}

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesos);

    // Colores de los comunicadores
    int carga = 0;
    int cota = 1;

    MPI_Comm_split(MPI_COMM_WORLD, carga, idProceso, &comunicadorCarga);

    SIGUIENTE = (idProceso + 1 ) % numProcesos;
    ANTERIOR = (idProceso - 1 + numProcesos) % numProcesos;

    int **matriz = reservarMatrizCuadrada(NCIUDADES);
    tNodo nodo,   // nodo a explorar
        lnodo,    // hijo izquierdo
        rnodo,    // hijo derecho
        solucion; // mejor solucion
    bool nueva_U,  // hay nuevo valor de c.s.
        fin;
    int U;        // valor de c.s.
    int iteraciones = 0;
    tPila pila; // pila de nodos a explorar

    U = INFINITO;    // inicializa cota superior
    InicNodo(&nodo); // inicializa estructura nodo

    if(idProceso == 0) {
        LeerMatriz(argv[2], matriz);
    }
    
    MPI_Bcast(&matriz[0][0], NCIUDADES*NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();

    if(idProceso != 0) {
        EquilibrarCarga(pila, fin);
        if(!fin)
            pila.pop(nodo);
    }

    fin = Inconsistente(matriz);

    while(!fin) {
        Ramifica(&nodo, &lnodo, &rnodo, matriz);
        nueva_U = false;

        if(Solucion(&rnodo)) {
            if(rnodo.ci() < U) {
                U = rnodo.ci();
                nueva_U = true;
            }
        } else {
            if(rnodo.ci() < U){
				pila.push(rnodo);
			}
        }

        if(Solucion(&lnodo)) {
            if(lnodo.ci() < U) {
                U = lnodo.ci();
                nueva_U = true;
            }
        } else {
            if(lnodo.ci() < U){
				pila.push(lnodo);
			}
        }

        if(nueva_U){
			pila.acotar(U);
		}
           
        EquilibrarCarga(pila, fin);
        if(!fin)
            pila.pop(nodo);
        iteraciones++;
        std::cout << idProceso << "-> " << iteraciones << " iteraciones." << std::endl;
    }
	
    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime() - t;
	liberarMatriz(matriz);
    MPI_Finalize();
    return 0;
}