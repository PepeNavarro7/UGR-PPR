ALL: version1

version1: version1.cpp
	mpicxx -O3 -o version1 version1.cpp

do: version1
	mpirun -np 4 version1 5 100
	@echo ""

clean:
	rm version1