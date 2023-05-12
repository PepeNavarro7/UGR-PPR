#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <math.h> 
MPI_Datatype MPI_BLOQUE;

using namespace std;

int main(int argc, char * argv[]) {
//////////////////////////////////////////////////////////////////////////////////
// Inicializamos valores
//////////////////////////////////////////////////////////////////////////////////
    int numeroProcesadores, id_Proceso;
    float *matriz_global, // Matriz global a multiplicar (solo util en 0)
	    *vector_multiplicacion, // Vector a multiplicar (solo util en 0)
        *resultado, // Vector resultado (solo util en 0)
        *multiplicacion_local, // Trozo local del vector de multiplicacion
        *resultado_local, // resultado local de cada submatriz
        *reduccion_local; // reduccion por filas

    double tInicio, // Tiempo en el que comienza la ejecucion
        Tpar, // Tiempo paralelo
        Tseq; // Tiempo secuencial

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesadores);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_Proceso);

    if (argc <= 1) {
        if (id_Proceso==0)
            cout << "Debes aportar N en la linea de comandos."<< endl;
        MPI_Finalize();
        return 0;
    } 
    const float raiz = sqrt(numeroProcesadores);
    if(raiz - floor(raiz) != 0.0){
        if (id_Proceso==0)
            cout << "El numero de procesos debe ser un cuadradado perfecto."<< endl;
        MPI_Finalize();
        return 0;
    }
    const int RAIZ_P = (int)raiz,
        N = atoi(argv[1]); // Tamaño del lado de la matriz grande
    if (N%numeroProcesadores != 0){
        if (id_Proceso==0)
            cout << "N debe ser multiplo de np."<< endl;
        MPI_Finalize();
        return 0;
    } 
    const int TAM = N/RAIZ_P, // Tamaño del lado de las submatrices
    FILA_P = id_Proceso/RAIZ_P, // Fila del proceso respecto al grid
    COLUMNA_P = id_Proceso%RAIZ_P; // Columna del proceso respecto al grid
    multiplicacion_local = new float[TAM]; // reservamos para el trozo del vector multiplicacion
    resultado_local = new float[TAM];
    reduccion_local = new float[TAM];
    

    // Proceso 0 genera matriz A y vector multiplicacion
    if (id_Proceso==0){
        matriz_global = new float[N*N]; //reservamos espacio para la matriz (N x N floats)
        resultado = new float[N]; //reservamos espacio para el vector resultado final (N floats)
        vector_multiplicacion = new float[N]; //reservamos espacio para el vector de la multiplicacion (N floats)

        // Rellena la matriz y el vector
        for (int i = 0; i < N; i++) {
            //vector_multiplicacion[i] = (float) (1.5*(1+(5*(i))%3)/(1+(i)%5));
            vector_multiplicacion[i] = 1;
            for (int j = 0; j < N; j++) {
	            //matriz_global[i*N+j] = (float) (1.5*(1+(5*(i+j))%3)/(1+(i+j)%5));
                matriz_global[i*N+j] = i*N+j;
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////////////
    // EMPAQUETAMOS DATOS Y ENVIAMOS
    //////////////////////////////////////////////////////////////////////////////////

    // Creo buffer de envío para almacenar los datos empaquetados
    float * buf_envio;
    if (id_Proceso==0){
        buf_envio=new float[N*N];
        // Defino el tipo bloque cuadrado 
        MPI_Type_vector (TAM, TAM, N, MPI_FLOAT, &MPI_BLOQUE);
        // Creo el nuevo tipo 
        MPI_Type_commit (&MPI_BLOQUE);
        // Empaqueta bloque a bloque en el buffer de envío
        int fila_P, columna_P, comienzo;
        for (int i=0, posicion=0; i<numeroProcesadores; i++){
            // Calculo la posicion de comienzo de cada submatriz 
            fila_P=i/RAIZ_P; // division entera
            columna_P=i%RAIZ_P;
            comienzo=(columna_P*TAM)+(fila_P*TAM*TAM*RAIZ_P);
            MPI_Pack (&matriz_global[comienzo], 1, MPI_BLOQUE, 
            buf_envio,sizeof(float)*N*N, &posicion, MPI_COMM_WORLD);
        }
        // Libero el tipo bloque
        MPI_Type_free (&MPI_BLOQUE);
    }
    // Creo un buffer de recepcion
    float *buf_recep = new float[TAM*TAM]; // Bufer de recepcion local
    // Distribuimos la matriz entre los procesos 
    MPI_Scatter (buf_envio, sizeof(float)*TAM*TAM, MPI_PACKED,buf_recep, TAM*TAM, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Ahora tenemos cada proceso con su submatriz en buf_recep
    MPI_Comm diagonal_com, // Comunicador de la diagonal
        columnas_com, // Comunicador por columnas
        filas_com; // Comunicador por filas
    int color_dia = (FILA_P==COLUMNA_P)?0:1;
    MPI_Comm_split(MPI_COMM_WORLD, COLUMNA_P, id_Proceso, &columnas_com);
    MPI_Comm_split(MPI_COMM_WORLD, color_dia, id_Proceso, &diagonal_com); 
    MPI_Comm_split(MPI_COMM_WORLD, FILA_P, id_Proceso, &filas_com); 

    // Repartimos el vector multiplicacion
    if (color_dia == 0){
        MPI_Scatter (vector_multiplicacion, TAM, MPI_FLOAT, 
                multiplicacion_local, TAM, MPI_FLOAT, 
                0, diagonal_com);
    }
    // Y ahora lo difundimos
    // El proceso raiz es el que tiene un id local (dentro de ese comunicador) que coincide con su numero de columna
    MPI_Bcast(multiplicacion_local, TAM, MPI_FLOAT, 
        COLUMNA_P, columnas_com); 

    //////////////////////////////////////////////////////////////////////////////////
    // CÁLCULO Y REDUCCION
    //////////////////////////////////////////////////////////////////////////////////

    // Hacemos una barrera para asegurar que todas los procesos comiencen la ejecucion
    // a la vez, para tener mejor control del tiempo empleado
    MPI_Barrier(MPI_COMM_WORLD);
    // Inicio de medicion de tiempo
    tInicio = MPI_Wtime();
    for (int i = 0; i < TAM; i++) {
        resultado_local[i]=0.0;
        for (int j = 0; j < TAM; j++) {
            resultado_local[i] += buf_recep[i*TAM+j] * multiplicacion_local[j];
        }
    }

    // Ahora cada proceso tiene un vector de tamaño TAM con la suma, hay que hacer una reduccion
    // todos los procesos de una misma fila reducirán sus vectores a uno, que tendrá el proceso diagonal
    MPI_Reduce (resultado_local, reduccion_local, TAM, MPI_FLOAT, MPI_SUM, FILA_P, filas_com);
    
    MPI_Barrier(MPI_COMM_WORLD);
    // fin de medicion de tiempo
    Tpar = MPI_Wtime()-tInicio;

    // Ahora simplemente juntamos los vectores reduccion en el proceso 0
    if (color_dia == 0){
        MPI_Gather (reduccion_local, TAM, MPI_FLOAT, 
                resultado, TAM, MPI_FLOAT, 
                0, diagonal_com);
    }

    // Terminamos la ejecucion de los procesos, despues de esto solo existira el proceso 0
    MPI_Finalize();

    //////////////////////////////////////////////////////////////////////////////////
    // CÁLCULO SECUENCIAL
    //////////////////////////////////////////////////////////////////////////////////
    if (id_Proceso == 0) {
        float * comprueba = new float [N];

        tInicio = MPI_Wtime();
        for (int i = 0; i < N; i++) {
	        comprueba[i] = 0.0;
	        for (int j = 0; j < N; j++) {
	            comprueba[i] += matriz_global[i*N+j] * vector_multiplicacion[j];
	        }
        }
        Tseq = MPI_Wtime()-tInicio;

        int errores = 0;
        for (int i = 0; i < N; i++) {   
            if (comprueba[i] != resultado[i])
                errores++;
        }

        // Liberamos memoria de proceso 0
        delete[] matriz_global;
        delete[] resultado;
        delete[] vector_multiplicacion;
        delete[] buf_envio;
        delete[] comprueba;
         

        //cout << "Se encontraron " << errores << " errores." << endl;
        //cout << "(sin contar el scatter y el broadcast)" << endl;
        cout << "Tiempo paralelo -----> " << Tpar*1000.0 << " milisegundos." << endl;
        //cout << "Tiempo secuencial ---> " << Tseq*1000.0 << " milisegundos." << endl;
    }

    // Liberamos memorias locales
    delete[] multiplicacion_local;
    delete[] resultado_local;
    delete[] reduccion_local;
    delete[] buf_recep;

    return 0;
}  
