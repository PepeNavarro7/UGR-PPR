#!/bin/bash

touch resultados.txt

for j in {1..3}
do
	echo "vuelta" $j >> resultados.txt
	for i in 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 2000000 5000000 10000000
	do
		echo $i >> resultados.txt
		./suma_vectores_reduction $i >> resultados.txt
	done
done

