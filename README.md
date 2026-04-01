# pruebas_tfm

## Exploración de datos

En el arcihvo `darwin_exp0.ipynb` se presenta un análisis exploratorio incial realizado al conjunto de datos de la base de datos DARWIN (en su totalidad).

En el archivo `darwin_exp1.ipynb` se presenta un análisis descriptivo y estadístico agrupado según la tarea indicada (1-25), para estudiar las variables mas influyentes y su relación según la tarea.

## Clasificadores

1. Regresion logistica (LR) 
2. XGBoost
3. Decision Tree (DT) 
4. Random Forest (RF) 
5. k-Nearest Neighbor (kNN)
6. Linear discriminant analysis (LDA)
7. Support Vector Machines (SVM)
8. Gaussian Naive Bayes (GNB)
9. Multilayer Perceptron (MLP)
10. Learning Vector Quantization (LVQ)

## Algoritmos de ML

El archivo `darwin2.py` se evalúan los 9 algoritmos BASE (sin mejora) sobre las 25 tareas del dataset DARWIN
y devuelve una tabla con Accuracy, Sensibilidad, Especificidad y F1 por tarea/algoritmo.

En el archivo `darwin_ml1.ipynb` se realiza una prueba de eliminación de variables, de aquellas con una alta correlación r > 0.90.

En el archivo `darwin_ml0.ipynb` se aplican los 10 clasificadores al conjunto de datos entero. Cada algoritmo se aplica dos veces, una primera vez con los parámetros por defecto y una segunda vez con los hiperparámetros definidos segín Cilia et al. (2022).

En el archivo `darwin_ml2.ipynb` se aplican los 10 lasificadores a cada tarea, por separado. Cada algoritmo se aplica dos veces, una primera vez con los parámetros por defecto y una segunda vez con los hiperparámetros definidos segín Cilia et al. (2022).

# LLM (SHAP y LIME)

Los cuatro algoritmos que mejores resultados arrojaron fueron Decision Tree (DT), Random Forest (RF), XGBoost y Gaussian Naive Bayes (GNB)

En el archivo `darwin_ml3.ipynb` se aplican los algoritmos de ML seleccionados (DT, RF, XGB y GNB) en su versión por defecto, al dataset entero, para aplicar a posteriori los LLM SHAP y LIME a cada ML.

En el archivo `darwin_ml4.ipynb` se aplican los algoritmos de ML seleccionados (DT, RF, XGB y GNB) en su versión por defecto a cada tarea del dataset, para aplicar a posterior los LLM SHAP y LIME a cada ML.

# NOTAS

- Al eliminar `mean_jerk_in_air` y `mean_speed_on_paper` por su alta correlación, los resultados no mejoran se mantienen iguales o peores.
- Los ML que mejor resultados arrojan son DT, RF, XGB y GNB.
- La tarea 19 parece ser la que mayor valor de Accuracy arroja al aplicar los disntos algoritmos de ML. Seguida va la tarea 7.
- Las variables mas importantes extraídas del LLM SHAP (del XGBoost aplicado al dataset entero) son `total_time23` y `air_time_17`.
- Al analizar el dataset por tareas con SHAP las variables mas destacadas en los distintos ML son `pressure_mean1`, `mean_speed_on_paper1`,

**Referencias bibliográficas**

Fontanella, F. (2022). DARWIN [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55D0K.

N. D.  Cilia,  C.  De  Stefano,  F.  Fontanella,  A.  S.  Di  Freca,  An experimental protocol to support cognitive impairment diagnosis by using handwriting analysis, Procedia Computer Science 141 (2018) 466–471.
https://doi.org/10.1016/j.procs.2018.10.141

N. D. Cilia, G. De Gregorio, C. De Stefano, F. Fontanella, A.  Marcelli, A. Parziale, Diagnosing Alzheimer’s disease from online handwriting: A novel dataset and performance benchmarking, Engineering Applications of Artificial Intelligence, Vol. 111 (2022) 104822.  
https://doi.org/10.1016/j.engappai.2022.104822
