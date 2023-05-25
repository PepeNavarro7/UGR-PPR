bbpar: bbpar.o libbb.o
	mpicxx -O3 bbpar.o libbb.o -o bbpar

bbseq: bbseq.o libbb.o
	mpicxx -O3 bbseq.o libbb.o -o bbseq
	
bbseq.o: bbseq.cc
	mpicxx -O3 -c bbseq.cc

bbpar.o: bbpar.cc
	mpicxx -O3 -c bbpar.cc

libbb.o: libbb.cc libbb.h
	mpicxx -O3 -c  libbb.cc 

do: bbpar
	mpirun -np 4 bbpar 4 ./tsp_problems/tsp4.1

clean:
	/bin/rm -f *.o bbseq bbpar



