ALL: version1 version2

version1: version1.cpp
	mpicxx -O3 -o version1 version1.cpp

version2: version2.cpp
	mpicxx -O3 -o version2 version2.cpp

do: version1
	mpirun -np 4 version1 5 100

clean:
	rm version1 version2