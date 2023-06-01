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
int rank, size;
MPI_Comm comunicadorCarga;	// Para la distribuci�n de la carga
MPI_Comm comunicadorCota; // Para la difusi�n de una nueva cota superior detectada

int main (int argc, char **argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Comm_split(MPI_COMM_WORLD,0,rank,&comunicadorCarga);
	MPI_Comm_split(MPI_COMM_WORLD,0,rank,&comunicadorCota);
	
	if(argc!=3 && rank==0){
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

	if(rank==0){
		LeerMatriz (argv[2], matriz);    // lee matriz de fichero
	}
	// Repartimos la matriz a todos los procesos
	MPI_Bcast(&matriz[0][0],NCIUDADES*NCIUDADES,MPI_INT,0,MPI_COMM_WORLD);

	if (rank != 0) {
		Equilibrar_Carga (&pila, &fin);
		if (!fin) 
			pila.pop(nodo);
	}
	
	/*while (! fin) { // ciclo del Branch&Bound
		Ramifica (&nodo, &nodo_izq, &nodo_dch);
			if (Solucion (&nodo_dch)) {
				if (ci (nodo_dch) < U) 
					U = ci (nodo_dch); // actualiza c.s.
			}
		else { // no es un nodo hoja
			if (ci (nodo_dch) < U) 
				Push (&pila, &nodo_dch);
		}
		if (Solucion(&nodo_izq)) {
			if (ci (nodo_izq) < U) 
				U = ci (nodo_izq); // actualiza c.s.
		}
		else { // no es nodo hoja
			if (ci (nodo_izq) < U) 
				Push (&pila, &nodo_izq);
		}
		Difusion_Cota_Superior (&U);
		if (hay_nueva_cota_superior) 
			Acotar (&pila, U);
		Equilibrado_Carga (&pila, &fin);
		if (! fin) 
			Pop (&pila, &nodo);
	}*/

	
	


/*

	activo = !Inconsistente(matriz); // Se examina si la matriz es correcta (tiene solucion o no)
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
		printf ("Solucion: \n");
		EscribeNodo(&solucion);
        cout << id_Proceso << "Tiempo gastado= "<<t<<endl;
		cout << "Numero de iteraciones = " << iteraciones << endl << endl;
	*/
    MPI_Finalize();
	if(rank==0){
		
		liberarMatriz(matriz);
	}
}