#!/bin/bash

touch resultados.txt
touch resultados_mod.txt

for i in 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
do
	echo $i >> resultados.txt
	for j in {1..5}
	do
		./suma_vectores_reduction $i >> resultados.txt
	done

	echo $i >> resultados_mod.txt
	for k in {1..5}
	do
		./suma_vectores_reduction_mod $i >> resultados_mod.txt
	done
done

