#!/bin/bash

touch resultados.txt

# Hacemos 3 vueltas
for i in 64 128 256
do
	echo "Tamanio de bloque -> " $i >> resultados.txt
	for j in 20000 30000 50000 75000 100000
	do 
		./transformacion $j $i >> resultados.txt
	done
done

