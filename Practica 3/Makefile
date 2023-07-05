bbpar: bbpar.o libbb.o
	mpicxx -O3 bbpar.o libbb.o -o bbpar

bbpar.o: bbpar.cc
	mpicxx -O3 -c bbpar.cc

libbb.o: libbb.cc libbb.h
	mpicxx -O3 -c  libbb.cc 

do: bbpar
	mpirun -np 4 bbpar 40 ./tsp_problems/tsp40.1

clean:
	/bin/rm -f *.o bbseq bbpar



