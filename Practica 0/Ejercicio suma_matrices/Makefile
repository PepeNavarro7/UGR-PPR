#Makefile

# CUDA code generation flags

GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SMXX    := -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS   ?= $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SMXX)
#OBJ = suma_matrices 
#SRC = suma_matrices.cu
ALL: suma_matrices suma_invertida suma_filas suma_columnas 

suma_matrices: suma_matrices.cu
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS) suma_matrices.cu -o suma_matrices

suma_invertida: suma_invertida.cu
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS) suma_invertida.cu -o suma_invertida

suma_filas: suma_filas.cu
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS) suma_filas.cu -o suma_filas

suma_columnas: suma_columnas.cu
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS) suma_columnas.cu -o suma_columnas

clean: 
	rm suma_matrices suma_invertida suma_filas suma_columnas

