ALL: matriz_profesor unidimensional bidimensional
	
matriz_profesor: 
	mpicxx -O3 -o matriz_profesor matriz_profesor.cpp

unidimensional:
	mpicxx -O3 -o unidimensional unidimensional.cpp

bidimensional:
	mpicxx -O3 -o bidimensional bidimensional.cpp
	
do: ALL
	mpirun -np 4 bidimensional 800

clean:
	rm matriz_profesor unidimensional bidimensional

