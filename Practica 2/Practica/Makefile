ALL: matriz_profesor unidimensional bidimensional_base bidimensional
	
matriz_profesor: 
	mpicxx -O3 -o matriz_profesor matriz_profesor.cpp

unidimensional:
	mpicxx -O3 -o unidimensional unidimensional.cpp

bidimensional_base:
	mpicxx -O3 -o bidimensional_base bidimensional_base.cpp

bidimensional:
	mpicxx -O3 -o bidimensional bidimensional.cpp

clean:
	rm matriz_profesor unidimensional bidimensional_base bidimensional
	rm resultados_uni.txt resultados_bi.txt

