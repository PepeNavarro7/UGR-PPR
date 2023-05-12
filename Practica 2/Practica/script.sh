#!/bin/bash

touch resultados_uni.txt resultados_bi.txt

# Hacemos 3 vueltas

for i in 360 720 1080 1440 1800
do
    echo "N="$i >> resultados_uni.txt
    echo "N="$i >> resultados_bi.txt

    for j in 4 9
    do 
        echo "np="$j >> resultados_uni.txt
        mpirun -np $j --oversubscribe unidimensional $i >> resultados_uni.txt

        echo "np="$j >> resultados_bi.txt
        mpirun -np $j --oversubscribe bidimensional $i >> resultados_bi.txt
    done

    echo " " >> resultados_uni.txt
    echo " " >> resultados_bi.txt
done
