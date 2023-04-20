#!/bin/bash

touch resultados.txt

# Hacemos 3 vueltas
for i in 64 256 1024
do
	echo "Tamanio de bloque -> " $i >> resultados.txt
	#for j in {1..3}
	#do
		#echo "Vuelta -> " $j >> resultados.txt
		./floyd $i ./input/input400 >> resultados.txt
		./floyd $i ./input/input1000 >> resultados.txt
		./floyd $i ./input/input1400 >> resultados.txt
		./floyd $i ./input/input2000 >> resultados.txt
		echo " " >> resultados.txt
	#done
done

