# pruebas_tfm

## Exploración de datos

En el arcihvo `darwin_exp0.ipynb` se presenta un análisis exploratorio incial realizado al conjunto de datos de la base de datos DARWIN (en su totalidad).

En el archivo `darwin_exp1.ipynb` se presenta un análisis descriptivo y estadístico agrupado según la tarea indicada (1-25), para estudiar las variables mas influyentes y su relación según la tarea.

El archivo `darwin2.py` simplemnte se encuentra parea realizar distintas pruebas y verificar posibles resultados, antes de efectuarlos en los notebooks

## Clasificadores

1. Regresion logistica (LR) 
2. Decision Tree (DT) 
3. Random Forest (RF) 
4. k-Nearest Neighbor (kNN)
5. Linear discriminant analysis (LDA)
6. Support Vector Machines (SVM)
7. Gaussian Naive Bayes (GNB)
8. Multilayer Perceptron (MLP)
9. Learning Vector Quantization (LVQ)

En el archivo `darwin_ml0.ipynb` se aplican los distintos clasificadores al conjunto de datos entero.

En el archivo `darwin_ml2.ipynb` se aplican los clasificadores a cada tarea, por separado.

En el archivo `darwin_ml1.ipynb` se realiza una prueba de eliminación de variables, de aquellas con una alta correlación r > 0.90.


**Referencias bibliográficas**

Fontanella, F. (2022). DARWIN [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55D0K.

N. D.  Cilia,  C.  De  Stefano,  F.  Fontanella,  A.  S.  Di  Freca,  An experimental protocol to support cognitive impairment diagnosis by using handwriting analysis, Procedia Computer Science 141 (2018) 466–471.
https://doi.org/10.1016/j.procs.2018.10.141

N. D. Cilia, G. De Gregorio, C. De Stefano, F. Fontanella, A.  Marcelli, A. Parziale, Diagnosing Alzheimer’s disease from online handwriting: A novel dataset and performance benchmarking, Engineering Applications of Artificial Intelligence, Vol. 111 (20229) 104822.  
https://doi.org/10.1016/j.engappai.2022.104822
