bbseq: bbseq.o libbb.o
	mpicxx -O3 bbseq.o libbb.o -o bbseq
	
bbseq.o: bbseq.cc
	mpicxx -O3 -c bbseq.cc


libbb.o: libbb.cc libbb.h
	mpicxx -O3 -c  libbb.cc 

do: bbseq
	mpirun -np 1 bbseq 4 ./tsp_problems/tsp4.1

clean:
	/bin/rm -f *.o bbseq



