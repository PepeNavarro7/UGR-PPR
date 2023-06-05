/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

//using namespace std;

unsigned int NCIUDADES;
int idProceso, numProcesos, ANTERIOR, SIGUIENTE;
MPI_Comm comunicadorCarga;	// Para la distribuci�n de la carga
MPI_Comm comunicadorCota; // Para la difusi�n de una nueva cota superior detectada

int main (int argc, char **argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesos);
    MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);
	SIGUIENTE = (idProceso+1)%numProcesos;
	ANTERIOR = (idProceso-1+numProcesos)%numProcesos;

	MPI_Comm_split(MPI_COMM_WORLD,0,idProceso,&comunicadorCarga);
	MPI_Comm_split(MPI_COMM_WORLD,0,idProceso,&comunicadorCota);
	
	if(argc!=3 && idProceso==0){
		std::cerr << "La sintaxis es: bbpar <tama�o> <archivo>" << std::endl;
		exit(1);
	}
	NCIUDADES = atoi(argv[1]);

	tNodo	nodo,         // nodo a explorar
			lnodo,        // hijo izquierdo
			rnodo,        // hijo derecho
			solucion;     // mejor solucion
	bool fin = false,        // condicion de fin
		nueva_U;       // hay nuevo valor de cota superior
	int  U;             // valor de cota superior
	int iteraciones = 0;
	tPila pila;         // pila de nodos a explorar

	int** matriz = reservarMatrizCuadrada(NCIUDADES); // matriz que guardará los datos
	U = INFINITO;                  // inicializa cota superior
	InicNodo (&nodo);              // inicializa estructura nodo

	if(idProceso==0){
		LeerMatriz (argv[2], matriz);    // lee matriz de fichero
	}
	// Repartimos la matriz a todos los procesos
	MPI_Bcast(&matriz[0][0],NCIUDADES*NCIUDADES,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (idProceso != 0) {
		Equilibrar_Carga (&pila, fin);
		if (!fin) 
			pila.pop(nodo); // saco un nodo de la pila y lo guardo en "nodo"
	}

	fin= Inconsistente(matriz);
	
	while (!fin) { // ciclo del Branch&Bound
		Ramifica (&nodo, &lnodo, &rnodo, matriz);
		if (Solucion (&rnodo)) {
			if (rnodo.ci() < U)  {
				U = rnodo.ci(); // actualiza c.s.
				nueva_U = true;
			}
		} else { // no es un nodo hoja
			if (rnodo.ci() < U) {
				pila.push(rnodo);
			}
		}
		if (Solucion(&lnodo)) {
			if (lnodo.ci() < U){
				U = lnodo.ci(); // actualiza c.s.
				nueva_U = true;
			}
		} else { // no es nodo hoja
			if (lnodo.ci() < U) 
				pila.push(lnodo);
		}
		if(nueva_U){
			pila.acotar(U);
			nueva_U=false;
		}
		//Difusion_Cota_Superior (&U);
		//if (hay_nueva_cota_superior) 
		//	Acotar (&pila, U);
		Equilibrar_Carga(&pila, fin);
		if (!fin)
			pila.pop(nodo);
		iteraciones++;
		std::cout << idProceso << "-> " << iteraciones << std::endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	std::cout << "Soy " << idProceso << " e hice " << iteraciones << " iteraciones." << std::endl;
    MPI_Finalize();
	if(idProceso==0){
		liberarMatriz(matriz);
	}
}